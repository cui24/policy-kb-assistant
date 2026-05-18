# RAG 评测集标注与评分规范（v1）

## 1. 目标
- 用统一标准构建企业政策知识库测试集，支持多入库策略公平对比。
- 区分三层能力：检索命中、答案质量、系统稳定性。

## 2. 文件与字段
评测模板文件：
- `data/eval/rag_eval_set_v2_100_template.csv`

字段说明：
- `query_id`: 题目唯一 ID（如 `Q_HR_001`）。
- `query`: 用户问题文本。
- `category`: `hr/admin/finance/it`。
- `difficulty`: `easy/medium/hard`。
- `query_type`: `fact/procedure/constraint/comparison/exception/multi_hop`。
- `gold_doc_id`: 权威答案文档 ID（单值）。
- `gold_section`: 权威证据所在章节/条款（可写“第三章 > 第十二条”）。
- `answer_points`: 关键答案点，建议 3-6 个，分号分隔。
- `as_of_date`: 时间敏感题的生效日期（如 `2026-05-01`），非时间题可空。
- `conflict_case`: 是否冲突题（`yes/no`）。
- `notes`: 备注（如“同主题多版本冲突”）。

## 3. 出题来源原则
- 80% 文档驱动：目录、条款、审批流程、额度/时限/例外规则。
- 20% 对抗题：模糊问法、跨段整合、冲突版本、缺条件问题。
- 不根据系统检索结果反向出题，避免“为了高分而出题”。

## 4. 标注原则
- `gold_doc_id` 只标一个“最权威依据”，不标候选集合。
- `gold_section` 必须可追溯到原文结构（章/节/条）。
- `answer_points` 只写可核验事实，不写抽象总结。
- 若多文档冲突，按优先级判定：
  1. `status=active` 优先于 `archived`
  2. 同家族版本高者优先（`version` 更高）
  3. `authority_level` 更高者优先（当前默认 `company_policy`）

## 5. 冲突题设计（建议 10-15 题）
典型场景：
- 同一制度不同版本（旧版 vs 新版）
- 同主题不同部门制度（口径不一致）
- “截至某日期”前后规则变化

冲突题标注要求：
- `conflict_case=yes`
- 必填 `as_of_date`
- `notes` 写明冲突来源（例：`FIN-02 v1.0 vs FIN-02 v2.0`）

## 6. 指标与通过标准
### 6.1 检索层（必须）
- `Recall@3`
- `Recall@5`
- `MRR`
- `GoldDoc Hit@5`

建议初始门槛（可按实际再调）：
- `GoldDoc Hit@5 >= 0.85`
- `MRR >= 0.65`

### 6.2 生成层（建议）
- `Answer Point Coverage`: 命中的 `answer_points` 比例
- `Faithfulness`: 是否由检索证据支持，是否幻觉
- `Conflict Handling`: 冲突题是否选对生效规则
- `Refusal Quality`: 证据不足时是否保守回复

建议评分（人工抽样）：
- 每题 0-2 分：`0=错误/幻觉`, `1=部分正确`, `2=完整且有依据`

### 6.3 系统层（建议）
- `Latency p50/p95`
- 错误率（`5xx`、超时）
- 工具成功率

## 7. 对比实验规范
- 同一批文档、同一套测试题、同一模型参数。
- 只改一个变量：切块/入库策略。
- 推荐 3 库对比：
  - `enterprise_kb_baseline_v1`
  - `enterprise_kb_overlap_v1`
  - `enterprise_kb_structured_v1`

## 8. 标注执行流程（实操）
1. 先填 `query`（100 条）并自检去重。
2. 填 `gold_doc_id` 与 `gold_section`。
3. 填 `answer_points`（3-6 点）。
4. 标记 `conflict_case` 与 `as_of_date`。
5. 二次抽查：每类至少复核 5 题。
6. 冻结评测集后再跑 3 套入库对比。
