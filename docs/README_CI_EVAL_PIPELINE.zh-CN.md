# 轻量 CI / 自动化评测流水线

## 1. CI 是什么

CI 是 `Continuous Integration`，持续集成。

在本项目里，它的作用是：

> 每次提交代码后，用固定环境自动跑测试和轻量评测，尽早发现业务逻辑、Agent 路由、记忆机制和质量指标是否退化。

CI 不是证明系统绝对正确，而是把关键能力变成可重复执行的检查。

## 2. 当前实现

当前 CI 分两层：

```text
ci-test
  -> 语法/导入检查
  -> 记忆测试
  -> Agent Graph L0 测试
  -> MCP wrapper 测试
  -> 关键记忆回归测试

ci-eval
  -> RAG fixture 指标计算
  -> Agent workflow 小样本评测
  -> 阈值门禁
  -> 输出 JSON/Markdown 报告
```

本地入口：

- `Makefile::ci-local`
- `Makefile::ci-test`
- `Makefile::ci-eval`

GitHub Actions：

- `.github/workflows/ci.yml`

质量门禁脚本：

- `scripts/ci_quality_gate.py`

输出目录：

- `outputs/ci/`

## 3. 为什么用 Docker 跑

这个项目本身是 Docker Compose 服务：

- API
- UI
- Postgres
- Redis
- Qdrant
- MCP
- KB init

因此 CI 也使用 Docker Compose 跑，避免出现：

```text
宿主机 Python 能跑
容器内服务不能跑
```

本地执行：

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d --build
make ci-local
```

或者分开跑：

```bash
make ci-test
make ci-eval
```

## 4. ci-test 逻辑

`ci-test` 负责确定性测试。

执行内容：

```bash
python -m compileall -q src tests scripts
pytest -q \
  tests/test_memory_l1_session.py \
  tests/test_agent_graph_working_memory.py \
  tests/test_agent_graph_mcp_client.py \
  tests/test_mcp_wrapper_executor.py \
  tests/test_services.py -k "memory or recent_draft or recent_ticket"
```

覆盖能力：

| 测试 | 作用 |
| --- | --- |
| `compileall` | 检查语法和导入级错误 |
| `test_memory_l1_session.py` | L1 会话记忆、TTL、pending task、最近轮次 |
| `test_agent_graph_working_memory.py` | L0 工作记忆与审计摘要 |
| `test_agent_graph_mcp_client.py` | 远程 MCP 熔断逻辑 |
| `test_mcp_wrapper_executor.py` | MCP wrapper 工具执行、幂等、错误归一 |
| `test_services.py -k memory...` | 旧版记忆路径兼容回归 |

这些测试主要验证：

- 业务规则是否稳定；
- 数据库状态是否正确；
- 权限/确认态/记忆边界是否正确；
- 不依赖真实 LLM 输出。

## 5. ci-eval 逻辑

`ci-eval` 负责轻量质量门禁。

执行：

```bash
python scripts/ci_quality_gate.py --out-dir outputs/ci
```

它分两部分。

### 5.1 RAG Fixture 指标

CI 里不跑真实 Qdrant/LLM 全量评测，而是用固定 fixture 验证指标计算和阈值门禁。

原因：

- 真实 RAG 全量评测慢；
- 依赖模型和向量库状态；
- CI 需要稳定、低成本、可重复。

RAG fixture 会计算：

| 指标 | 含义 |
| --- | --- |
| `GoldDoc Recall@3` | gold_doc_id 是否进入 Top3，文档级命中 |
| `GoldDoc Recall@5` | gold_doc_id 是否进入 Top5，文档级命中 |
| `GoldDoc MRR` | 正确文档排名的倒数平均 |
| `Auto APC` | 自动答案要点覆盖率 |
| `Citation Output Rate` | 引用输出率 |
| `Refusal Rate` | 拒答率 |
| `Retrieve p95` | 检索耗时 p95 |

口径注意：

- `GoldDoc Recall/MRR` 是文档级命中，不是严格条款级命中；
- `Auto APC` 是自动答案要点覆盖率，不是人工最终准确率；
- `Citation Output Rate` 只说明是否输出引用，不说明引用一定正确。

### 5.2 Agent Workflow 小样本评测

Agent 评测复用：

- `src/api/planner_eval.py`
- `data/agent/global_planner_regression_cases.jsonl`

CI 中使用 `rules` 策略和 patched services，避免依赖真实 LLM。

评测目标：

- 用户输入是否路由到正确业务方向；
- 无工单号时是否追问；
- 有工单号时是否进入工单工具；
- 草稿/查单/补充/催办/取消等路径是否保持稳定。

核心指标：

```text
Route Accuracy = route 匹配数 / 执行样例数
```

## 6. 阈值门禁

默认阈值在 `scripts/ci_quality_gate.py` 中：

```text
min_rag_r3 = 0.75
min_rag_mrr = 0.60
min_rag_apc = 0.70
min_citation_rate = 0.70
min_agent_route_accuracy = 0.80
agent_error_count = 0
```

如果任一指标低于阈值，脚本返回 `exit 1`，CI 失败。

## 7. 输出产物

`ci-eval` 会生成：

```text
outputs/ci/quality_gate_latest.json
outputs/ci/quality_gate_latest.md
outputs/ci/rag_fixture_latest.json
outputs/ci/agent_eval_latest.json
```

GitHub Actions 会上传 `outputs/ci/` 作为 artifact。

## 8. 和全量 RAG 评测的关系

轻量 CI 不替代全量 RAG 评测。

当前分工：

```text
每次 push / PR：
  跑 ci-test + ci-eval

重要改动或手动评测：
  跑 scripts/evaluate_policy_eval_set.py
  使用 data/eval/policy_eval_total_130.csv
  输出 outputs/eval_compare/*
```

全量评测关注真实质量：

- 混合检索；
- Rerank；
- 文档级 Recall/MRR；
- 自动答案要点覆盖；
- 引用输出；
- 延迟。

轻量 CI 关注防回归：

- 指标计算逻辑没坏；
- Agent 路由没明显坏；
- 记忆和工具链没坏。

## 9. 面试表述

> 我把 CI 分成确定性测试和轻量质量评测两层。确定性测试覆盖记忆、Agent Graph、MCP wrapper 和服务层回归；质量评测用固定 RAG fixture 验证指标计算和阈值门禁，同时复用 Agent workflow regression cases 评估路由稳定性。全量 RAG 评测不放在每次 CI 里，而是作为手动/阶段性评测，避免 CI 依赖真实模型和向量库状态。

