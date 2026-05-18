# RAG优化与评测统一总结（2026-05-14）

## 1. 问题描述
在企业制度 RAG 场景中，我们同时遇到两个目标冲突：
- 检索排序目标：希望正确证据尽量排到 Top1（高 `R@1/MRR`）。
- 生成回答目标：希望 TopK 证据集合尽量完整（高 `APC/Citation`、低 `Refusal`）。

三种策略（`fixed/overlap/structured_hybrid`）在不同目标上表现不一致，导致“哪种最好”不能用单一指标判断。

## 2. 问题原因
### 2.1 文档形态原因
企业制度天然是“章→条→款→项→流程/条件/例外/清单”的层级结构，不是普通连续文章。

### 2.2 切分机制原因
- `structured_hybrid` 保留层级结构，召回覆盖更强，但可能产生较多短块，候选间竞争加剧。
- `overlap` 保留连续上下文，单块信息更完整，Top1 排序更稳定。

### 2.3 指标偏好原因
- 检索指标（尤其 `R@1/MRR`）偏向“第一条是否命中”。
- 端到端生成更依赖 TopK 证据集合，而非单条 Top1。

## 3. 优化方法
### 3.1 切分策略优化
- 同条优先合并；同章相邻条按主题和长度约束合并。
- 超短块标题增强；超长块按子结构二次切分。
- 窗口化跨页切分，减少页边界语义断裂。

### 3.2 评测集与流程优化
- 构建困难检索子集（hard30）并与基础集合并为总集（130题）。
- 固定同一编码器（`bge-small`）比较三库，避免模型变量干扰。
- 分层评估检索与生成（retrieval + with-answer）。

### 3.3 检索链路优化：Hybrid Retrieval + RRF + Rerank
在原有 dense 向量检索基础上，新增两级检索增强：

1. **混合召回（Hybrid Retrieval）**
   - dense：使用 Qdrant 向量检索召回语义相近片段。
   - sparse：从 Qdrant payload 构建运行时 BM25 内存索引，按关键词和专有名词召回片段。
   - 中文 BM25 采用轻量 2/3-gram，英文/数字 token 保留完整，兼顾制度条款、编号、系统名等精确匹配。

2. **RRF 融合（Reciprocal Rank Fusion）**
   - dense 与 BM25 的原始分数尺度不同，不直接相加。
   - 采用 RRF 按排名融合：
     `score = dense_weight / (rrf_k + dense_rank) + bm25_weight / (rrf_k + bm25_rank)`。
   - 当前参数：`dense_candidate_k=40`、`bm25_candidate_k=40`、`rrf_k=60`、`dense_weight=1.0`、`bm25_weight=0.8`。

3. **CrossEncoder 重排序（Rerank）**
   - 对 RRF 融合后的候选片段，用 `BAAI/bge-reranker-base` 对 `(query, chunk_text)` 重新打分。
   - 当前参数：`rerank_candidate_k=30`、`batch_size=8`、`max_chars=1200`。
   - 该模式提升排序质量，但 CPU 容器内延迟较高，更适合作为高质量模式或离线评测基线。

## 4. 关键结果
### 4.1 总集130（检索）
| 策略 | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| fixed | 0.677 | 0.854 | 0.915 | 0.778 |
| overlap | 0.769 | 0.869 | 0.915 | 0.829 |
| structured_hybrid | 0.738 | 0.892 | 0.931 | 0.821 |

### 4.2 总集130（生成）
| 策略 | APC | Citation | Refusal |
|---|---:|---:|---:|
| fixed | 0.239 | 0.715 | 0.300 |
| overlap | 0.320 | 0.769 | 0.254 |
| structured_hybrid | 0.435 | 0.838 | 0.169 |

### 4.3 检索时延（同模型三库）
稳态（`p50/p95`）三库都在几十毫秒量级，差异不大。
`fixed` 在部分运行存在明显长尾离群值，`avg` 被拉高。

### 4.4 同策略跨模型对比（structured_hybrid，total130 实测）
基于 `policy_eval_total_130.csv` 的同策略跨模型实测结果：

| 模型 | R@1 | R@3 | R@5 | MRR | p50(ms) | p95(ms) |
|---|---:|---:|---:|---:|---:|---:|
| BAAI/bge-small-zh-v1.5 | 0.738 | 0.892 | 0.931 | 0.821 | 36 | 46 |
| BAAI/bge-base-zh-v1.5 | 0.715 | 0.877 | 0.931 | 0.805 | 83 | 117 |
| moka-ai/m3e-base | 0.731 | 0.862 | 0.885 | 0.805 | 80 | 108 |
| BAAI/bge-large-zh-v1.5 | 0.715 | 0.862 | 0.900 | 0.794 | 232 | 337 |

结论（同策略下）：
- `bge-small` 在 structured_hybrid 策略上准确率（R@1/R@3/MRR）最高，且检索时延最低。
- 在 8GB 显存场景，`bge-small` 的性能-成本比最佳。

### 4.5 新增检索链路优化对比（Docker API 容器，total130，端到端生成）
本轮在 Docker `api` 容器内，对同一评测集和同一 collection 做三组对比：

- 评测集：`data/eval/policy_eval_total_130.csv`
- Collection：`policy_kb_structured_hybrid__baai_bge-large-zh-v1.5_v1`
- Embedding：`BAAI/bge-large-zh-v1.5`
- TopK：`10`
- 生成：开启，使用容器环境中的 `deepseek-chat`

指标口径：
- `GoldDoc R@K` / `GoldDoc MRR` 是**文档级命中**，只判断 `gold_doc_id` 是否进入 TopK，不等同于严格条款级命中。
- `Auto APC` 是**自动答案要点覆盖率**，基于轻量字符串/词片段匹配，不等同于人工最终正确率。
- `Citation Output` 是**引用输出率**，只表示答案是否输出引用，不保证引用一定精确支持答案。

| 模式 | GoldDoc R@3 | GoldDoc R@5 | GoldDoc MRR | Auto APC | Citation Output | Refusal | Retrieve p50 | Retrieve p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 86.15% | 90.00% | 0.7955 | 40.63% | 86.15% | 14.62% | 211 ms | 308 ms |
| hybrid_rrf | 93.08% | 96.15% | 0.8574 | 43.31% | 88.46% | 13.08% | 216 ms | 339 ms |
| hybrid_rrf_rerank | 96.92% | 97.69% | 0.9164 | 47.64% | 93.08% | 6.92% | 11192 ms | 16168 ms |

相对 dense 的提升：
- `hybrid_rrf`：GoldDoc R@5 +6.15pp，GoldDoc MRR +0.0619，Auto APC +2.68pp，p50 检索延迟仅 +5ms。
- `hybrid_rrf_rerank`：GoldDoc R@5 +7.69pp，GoldDoc MRR +0.1210，Auto APC +7.01pp，拒答率 -7.69pp，但 p50 检索延迟增加约 11s。

保留结果：
- 汇总：`outputs/eval_compare/retrieval_mode_compare_full_20260514.md`
- 机器可读汇总：`outputs/eval_compare/retrieval_mode_compare_full_20260514.json`
- 三组逐题结果：
  - `outputs/eval_dense_full/`
  - `outputs/eval_hybrid_full/`
  - `outputs/eval_hybrid_rerank_full/`

### 4.6 最终固定推荐组合
主推荐（综合检索与生成质量）：
- **`structured_hybrid + BAAI/bge-small-zh-v1.5 + Qdrant(collection=policy_kb_structured_hybrid__baai_bge-small-zh-v1.5_v1)`**

备选（若你把 Top1 排序稳定性放在第一优先）：
- `overlap + BAAI/bge-small-zh-v1.5 + Qdrant(collection=policy_kb_overlap__baai_bge-small-zh-v1.5_v1)`

新增运行时检索策略建议：
- 默认实时模式：`hybrid_rrf`，因为质量提升明显且延迟接近 dense。
- 高质量/离线模式：`hybrid_rrf_rerank`，因为排序和生成质量最好，但 CPU 容器内延迟较高。

## 5. 评估结果分析（重点成因）
### 5.1 structured_hybrid 为什么召回更好
- 结构路径与条款边界保留后，更容易定位到正确章节与规则组。
- 候选片段主题更聚焦，因此 `R@3/R@5` 更高。

### 5.2 为什么 structured_hybrid 的排序不一定超过 overlap
- 它常召回多个“都相关”的短片段，内部竞争导致正确证据不一定排第1。
- 结果表现为：`TopK` 覆盖强，但 `R@1/MRR` 不一定最优。

### 5.3 overlap 为什么排序更强
- 连续上下文块更容易覆盖问题中的多个关键词共现。
- 向量相似度更容易把该块判为第一相关，因此 `R@1/MRR` 更稳。

### 5.4 为什么 structured_hybrid 的端到端生成质量最好
- 生成模型依赖 TopK 证据集合，而不是只看第1条。
- structured_hybrid 更容易把“条件+流程+材料+例外+审批角色”同时放进 TopK，
  因而 `APC/Citation` 更高、`Refusal` 更低。

### 5.5 为什么 Hybrid RRF 能提升结果
- dense 向量检索擅长语义相似，但对制度编号、专有名词、固定短语和关键词精确匹配不一定稳定。
- BM25 擅长词面匹配，能补上 dense 漏掉的编号、系统名、条款关键词和业务术语。
- RRF 只使用两路结果的排名，不要求 dense 分数和 BM25 分数处于同一尺度，因此比直接分数加权更稳。
- 本次结果中，`hybrid_rrf` 在几乎不增加检索 p50 延迟的情况下，把 GoldDoc R@5 从 90.00% 提升到 96.15%，说明 sparse 召回确实补上了部分 dense 漏召回样本。

### 5.6 为什么 Rerank 能继续提升，但延迟明显增加
- RRF 融合解决的是“候选召回更全”，但最终排序仍然只基于两路召回排名。
- CrossEncoder reranker 会把 `query` 和每个候选 `chunk_text` 放在一起做相关性判断，比单向量相似度更能识别“这个片段是否真正回答当前问题”。
- 因此 `hybrid_rrf_rerank` 的 GoldDoc MRR 从 0.8574 提升到 0.9164，Auto APC 从 43.31% 提升到 47.64%，拒答率从 13.08% 降到 6.92%。
- 代价是每题要对最多 30 个候选做 CrossEncoder 推理，CPU 容器内 p50 检索延迟达到 11192ms；因此不建议直接作为默认实时模式。

### 5.7 结论落地
- 默认工程检索方案：优先 `overlap`（排序稳定）。
- 回答质量优化方向：优先 `structured_hybrid`（证据集合质量高）。
- 在线检索默认策略：优先 `hybrid_rrf`（质量提升明显，延迟代价小）。
- 高质量评测/离线模式：使用 `hybrid_rrf_rerank`（排序和生成质量最好，但 CPU 延迟高）。
- 下一步：推进 `structured_pack` 与轻量 rerank/缓存/GPU 优化，减少短块竞争并降低重排序延迟。

## 6. 保留文件（核心）
- 总结文档：`docs/README_CHUNKING_EVAL_UNIFIED_20260514.zh-CN.md`
- 评测集：`data/eval/policy_eval_total_130.csv`
- 困难子集：`data/eval/policy_eval_hard_retrieval_30.csv`
- 检索总集结果：`outputs/policy_eval_total_130_three_kb/`
- 生成总集结果：`outputs/policy_eval_total_130_with_answer_three_kb/`
- 困难三库结果：`outputs/policy_eval_hard_retrieval_30_three_kb/`
- 时延汇总：`outputs/latency_summary/retrieval_latency_three_kb_bge_small.csv`
- 同策略跨模型（structured_hybrid, total130）：`outputs/latency_summary/structured_hybrid_model_compare_total130.csv`
- 新增检索链路对比：`outputs/eval_compare/retrieval_mode_compare_full_20260514.md`
- 新增检索链路机器可读汇总：`outputs/eval_compare/retrieval_mode_compare_full_20260514.json`
