# 监控与成本统计实现总结

本文档总结本窗口围绕“监控 / 成本统计”做过的设计和代码实现，重点说明：

1. 当前项目为什么需要这层能力。
2. ASK 问答链路的延迟、成功率、JSON 稳定性、token 和成本如何记录。
3. `/ops/metrics` 这类汇总接口的实现逻辑。
4. 工单监控为什么优先从审计表切入。
5. 当前仓库里需要继续确认的注意点。

## 1. 目标

原项目已经能完成政策问答和工单操作，但对“系统运行得怎么样”缺少结构化观测。

这轮优化的目标不是一次性接入 Prometheus、Grafana 或完整 APM，而是先做低风险版本：

- 每次 ASK 问答后，把关键运行指标落库。
- 能查询最近一段时间的问答质量、延迟和失败分布。
- 能估算 token 消耗和模型调用成本。
- 工单相关操作优先通过审计表统计成功和失败，而不是只看业务表。
- 保持实现简单，不改变主业务链路行为。

这对求职展示很有价值，因为它把项目从“功能 demo”推进到“可运营服务”的方向：不仅能回答问题，还能知道系统是否稳定、哪里慢、哪里贵、哪里失败。

## 2. 涉及文件

当前实现主要分布在以下文件：

- `src/kb/answer.py`：模型调用、JSON 修复、fallback、token usage 采集、成本估算。
- `src/api/ask_pipeline.py`：从模型返回的 `meta` 中提取 usage/cost，并写入 `kb_queries`。
- `src/api/services.py`：在异步、流式、历史详情等路径中传递和返回 usage/cost 字段。
- `src/api/models.py`：为 `KBQuery` ORM 增加 usage/cost 字段。
- `src/api/schemas.py`：为问答详情响应增加 usage/cost 字段。
- `alembic/versions/0013_add_kb_query_usage_metrics.py`：数据库迁移脚本。
- `src/api/ops_metrics.py`：ASK 监控指标聚合函数。
- `src/api/routes/ops.py`：提供 `/ops/metrics` 查询入口和 `pretty=true` 格式化输出。
- `tests/test_api_smoke.py`：补充 smoke test 断言，但当前环境缺少 `pytest`，未能实际运行。

## 3. 数据库字段

`kb_queries` 表新增了 6 个字段：

| 字段 | 含义 |
| --- | --- |
| `repair_used` | 本次回答是否触发 JSON 修复调用 |
| `prompt_tokens` | 输入 token 数 |
| `completion_tokens` | 输出 token 数 |
| `total_tokens` | 总 token 数 |
| `token_usage_estimated` | token 是否为本地估算值 |
| `estimated_cost_usd` | 按环境变量配置估算出的美元成本 |

迁移文件是：

```bash
alembic/versions/0013_add_kb_query_usage_metrics.py
```

迁移脚本使用 `_has_table()` 和 `_has_column()` 做保护，因此重复执行时更安全。API 启动时会通过 `src/api/app.py` 的 startup 事件调用 `ensure_schema_ready()`，正常容器启动会自动迁移。

如果需要手动执行迁移，建议在 API 容器内运行：

```bash
docker compose -f compose.yaml -f compose.dev.yaml exec -T api alembic upgrade head
```

通常不需要重新 build 镜像；如果只是 Python 代码和迁移文件变化，开发环境绑定了 `/app` 卷时，强制重建 API 容器即可：

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d --force-recreate --no-deps api
```

## 4. token 与成本采集逻辑

ASK 的核心回答逻辑在 `src/kb/answer.py`。

模型调用后，代码会优先从 OpenAI-compatible 响应对象中读取 `response.usage`：

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

如果模型服务商没有返回 usage，则使用字符数做粗略估算：

```text
粗估 token 数 = 文本字符数 / 2 后向上取整
```

这个估算不精确，但足够用于 demo 级成本趋势观察，并且会把 `token_usage_estimated` 标记为 `true`，避免把估算值误当成官方计量。

成本估算不在代码里硬编码模型价格，而是读取环境变量：

```bash
LLM_INPUT_COST_PER_1M_USD
LLM_OUTPUT_COST_PER_1M_USD
```

公式是：

```text
estimated_cost_usd =
  prompt_tokens / 1_000_000 * LLM_INPUT_COST_PER_1M_USD
  +
  completion_tokens / 1_000_000 * LLM_OUTPUT_COST_PER_1M_USD
```

这样做的原因是模型价格变化很快，价格应该由部署环境配置，而不是写死在业务代码里。

## 5. JSON 修复与 fallback 的 usage 合并

当前 ASK 链路不是单次模型调用，而是可能包含多次调用：

1. 主模型回答。
2. 主模型输出不完整时，增加 token 上限重试。
3. 输出无法解析或结构不合格时，调用修复模型做 JSON repair。
4. 仍失败时，切换 fallback 模型。
5. 最终仍失败时，返回保守拒答。

因此 usage 不能只记录最后一次调用，而是需要把多次模型调用累加。

`src/kb/answer.py` 里通过 `_merge_usage()` 合并多次调用的 token，用 `_attach_meta()` 把最终的运行信息挂到返回结果：

- `json_ok`
- `repair_used`
- `failure_reason`
- `attempt_stage`
- `usage`
- `token_usage_estimated`
- `estimated_cost_usd`

后续 API 层统一从 `meta` 里读取这些字段，避免每条业务路径重复理解模型调用细节。

## 6. ASK 落库逻辑

`src/api/ask_pipeline.py` 负责把模型输出变成可落库的 `KBQuery`。

关键步骤：

1. `run_retrieve_step()` 记录检索耗时。
2. `run_answer_step()` 记录回答耗时。
3. `normalize_answer_payload()` 归一化回答、引用、trace、耗时和 meta。
4. `usage_metrics_from_meta()` 从 `meta` 中提取 usage/cost。
5. `persist_kb_query()` 写入 `kb_queries`。

落库时会保存：

- 用户、部门、问题、答案。
- 引用和检索 top-k trace。
- `attempt_stage`。
- 检索耗时与回答耗时。
- 模型名。
- JSON 是否有效。
- 失败原因。
- JSON repair、token 和成本字段。

审计日志也会写入 usage 摘要，方便之后从 `audit_logs` 追踪单次请求。

## 7. ASK 监控聚合逻辑

`src/api/ops_metrics.py` 采用只读聚合方式：不调用模型、不访问向量库、不修改数据库，只查询 `kb_queries`。

核心统计流程：

1. 根据 `hours` 计算时间窗口，默认最近 24 小时，最大 7 天。
2. 查询 `created_at >= since` 的 `KBQuery` 记录。
3. 统计请求量、成功数、失败数。
4. 统计 JSON 有效率、引用率、repair 比例、fallback 比例。
5. 统计检索耗时、回答耗时、总耗时的平均值。
6. 使用 nearest-rank 方法计算 p50、p95 总耗时。
7. 输出失败原因分布、执行阶段分布、模型分布。

当前成功 / 失败定义：

- `failure_reason` 有值：视为失败。
- `failure_reason` 为空：视为成功。

这个定义比简单看 `valid_json=false` 更适合当前项目，因为有些保守拒答可能是系统设计内行为，不一定代表链路异常。

## 8. `/ops/metrics` 查询方式

路由文件是 `src/api/routes/ops.py`，设计入口为：

```bash
curl "http://localhost:8080/ops/metrics?hours=24"
```

为了方便人直接阅读，增加了 `pretty=true`：

```bash
curl "http://localhost:8080/ops/metrics?hours=24&pretty=true"
```

`pretty=true` 会返回带缩进和换行的 JSON，适合本地排查和面试演示。

## 9. 工单监控思路

工单监控推荐优先从 `audit_logs` 做，而不是只从 `tickets`、`ticket_drafts`、`pending_actions` 这些业务表做。

原因是：

- 业务表通常只保留成功后的最终状态。
- 失败的工具调用、被拒绝的操作、权限失败、参数不完整等过程，很多不会体现在最终业务表里。
- 审计表记录动作过程，更适合统计成功率、失败率、工具调用分布和拒绝原因。

低风险实现思路：

1. 从 `audit_logs` 统计 `action_type` 分布。
2. 从审计 payload 中统计 route、tool、rejection_reason。
3. 从业务表补充当前存量状态，例如工单总数、状态分布、优先级分布、草稿转化率。
4. 不改变工单执行逻辑，只增加只读聚合。

这样既能看到“系统做成了什么”，也能看到“系统哪里没做成”。

## 10. 当前仓库注意点

当前本地仓库中，`src/api/routes/ops.py` 导入的是：

```python
from src.api.ops_metrics import get_ops_metrics
```

但实际读到的 `src/api/ops_metrics.py` 里当前只有 `get_ask_metrics()`，没有 `get_ops_metrics()`。

这意味着如果当前文件状态直接启动 API，可能出现导入失败。后续需要二选一修正：

1. 在 `ops_metrics.py` 中补回 `get_ops_metrics()`，让它组合 ASK 指标和工单指标。
2. 或者临时把路由改成调用 `get_ask_metrics()`，只暴露 ASK 监控。

从项目演进角度，更推荐方案 1，因为后续 Redis 缓存命中率、工单监控、成本总览都可以统一挂在 `/ops/metrics` 下。

## 11. 后续扩展建议

短期推荐继续补三件事：

1. 在 `/ops/metrics` 中加入 token/cost 聚合：
   - `total_tokens`
   - `prompt_tokens`
   - `completion_tokens`
   - `estimated_cost_usd`
   - `estimated_usage_count`
2. 补齐工单监控：
   - action 分布
   - route 分布
   - tool 分布
   - rejection reason 分布
   - 工单状态、优先级、类别分布
3. 增加 Redis 问答结果缓存后，把缓存指标纳入监控：
   - cache hit count
   - cache miss count
   - cache hit rate
   - estimated saved tokens
   - estimated saved cost

中期再考虑接入 Prometheus/Grafana。当前阶段先把数据落库和结构化查询做好，更符合两天内可落地、低风险、可面试讲清楚的目标。
