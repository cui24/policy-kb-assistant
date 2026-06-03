# 并发测试结果总结

本文档记录当前项目的并发测试过程、关键结果和面试叙事结论。

## 1. 测试目标

本轮测试不是为了证明系统能无限扩容，而是为了回答三个问题：

1. 请求从入口进入后，哪一层最先成为瓶颈？
2. MCP 客户端并发调用工具时，瓶颈在 MCP 层、业务层、数据库，还是 LLM API？
3. 面试中如何讲清楚“我不仅做了功能，还验证过并发短板”？

## 2. 测试范围

已测试：

- FastAPI 入口层：`/health`
- FastAPI DB 读路径：`GET /tickets/{ticket_id}`
- FastAPI 评论写路径：`POST /tickets/{ticket_id}/comments`
- FastAPI 单工单热点更新：`POST /tickets/{ticket_id}/escalate`
- FastAPI 多工单分散更新：`escalate_many`
- FastAPI 建单写路径：`POST /tickets`
- MCP 工具层：`list_tools`
- MCP 工具层：`get_ticket_detail`

未混入本轮结论：

- LLM API 并发
- RAG 问答端到端并发
- Agent planner 端到端并发

原因是 LLM API 属于外部依赖，受模型服务限流、网络延迟、token 生成速度和费用影响。如果和 MCP/DB 压测混在一起，会干扰瓶颈定位。

## 3. 测试脚本

FastAPI 直连压测：

```bash
python scripts/concurrency_smoke.py --case health
python scripts/concurrency_smoke.py --case lookup
python scripts/concurrency_smoke.py --case comment
python scripts/concurrency_smoke.py --case escalate
python scripts/concurrency_smoke.py --case escalate_many
python scripts/concurrency_smoke.py --case create
```

MCP 工具层压测：

```bash
python scripts/mcp_concurrency_smoke.py --case health
python scripts/mcp_concurrency_smoke.py --case lookup
python scripts/mcp_concurrency_smoke.py --case create
```

宿主机如果没有安装 `mcp` Python 包，可以通过容器运行：

```bash
docker compose exec -T api python - \
  --case lookup \
  --requests 1000 \
  --concurrency 100 \
  --mcp-url http://mcp:9000/mcp \
  --api-base-url http://api:8080 \
  < scripts/mcp_concurrency_smoke.py
```

## 4. FastAPI 入口层

小规模 `/health` 测试通过：

```text
case=health
total=5
ok=5
error=0
p95≈16ms
```

结论：

- 入口层本身不是最先出现的瓶颈。
- 在服务健康时，普通探活响应很快。
- 后续高压写请求把服务拖到 unhealthy 时，`/health` 也会超时，原因不是 `/health` 自身复杂，而是同步线程池被阻塞请求占满。

## 5. FastAPI DB 读路径

小规模查单测试通过：

```text
case=lookup
total=5
ok=5
error=0
p95≈21ms
```

大并发查单测试：

```text
case=lookup
requests=1000
concurrency=100
ok=30
error=970
rps≈6.63
p50≈15075ms
p95≈15107ms
statuses={'200': 30, 'exception:ReadTimeout': 970}
```

结论：

- 普通 DB 读路径在低中并发下稳定。
- 100 并发直连 API 查单会大量超时。
- 查单链路包含鉴权、DB 查询、评论读取和响应序列化。
- 即使是读请求，也会先经过 JWT 鉴权查用户并创建 DB session。
- 高并发下仍然可能触发同步线程池和 DB 连接池排队。

## 6. 评论写路径

小规模并发评论测试通过：

```text
case=comment
total=5
ok=5
error=0
p95≈42ms
```

结论：

- append-only 评论写入在小并发下正常。
- 该路径比状态更新安全，因为主要是插入评论表，不会高频覆盖同一行状态字段。

## 7. 单工单热点更新

单工单并发催办成功样例：

```text
case=escalate
requests=500
concurrency=50
ok=500
error=0
rps≈100
p95≈951ms
p99≈1260ms
actual_delta=500
```

单工单并发催办失败样例：

```text
case=escalate
requests=600
concurrency=60
ok=17
error=583
statuses={'200': 17, 'exception:ReadTimeout': 583}
p50≈15029ms
```

结论：

- 50 并发热点更新稳定。
- 60 并发进入雪崩区间。
- `escalation_count` 校验通过时，说明成功请求没有丢更新。
- 失败原因不是业务校验错误，而是请求排队、DB 连接池耗尽和同步线程池阻塞。

关键日志：

```text
sqlalchemy.exc.TimeoutError:
QueuePool limit of size 5 overflow 10 reached,
connection timed out, timeout 30.00
```

## 8. 多工单分散更新

多工单分散写梯度测试：

| 场景 | 结果 | p95 | 一致性 |
|---|---:|---:|---|
| 20 并发 / 200 请求 / 20 工单 | 200/200 成功 | ≈226ms | 计数一致 |
| 30 并发 / 300 请求 / 20 工单 | 300/300 成功 | ≈498ms | 计数一致 |
| 40 并发 / 400 请求 / 20 工单 | 400/400 成功 | ≈845ms | 计数一致 |
| 50 并发 / 500 请求 / 20 工单 | 500/500 成功 | ≈988ms | 计数一致 |
| 60 并发 / 600 请求 / 20 工单 | 58/600 成功 | ≈15023ms | 压测后校验超时 |

结论：

- 分散到多张工单后，50 并发仍然稳定。
- 60 并发仍然失败，说明瓶颈不只是单行锁。
- 整体同步写路径、DB 连接池、审计日志写入和线程池阻塞共同形成瓶颈。

## 9. 建单写路径

小中并发建单测试：

```text
case=create
requests=20
concurrency=5
ok=20
error=0
p95≈113ms
unique_ticket_ids=20
```

```text
case=create
requests=100
concurrency=10
ok=100
error=0
p95≈202ms
unique_ticket_ids=100
```

高并发建单失败样例：

```text
case=create
requests=600
concurrency=60
ok=0
error=600
statuses={'exception:ReadTimeout': 600}
p50≈15052ms
p95≈15064ms
```

结论：

- 建单路径在低中并发下稳定。
- 50 并发用户侧反馈可通过。
- 60 并发进入超时区间。
- `ok=0` 只代表客户端 15 秒内没有收到响应，不一定代表数据库中 0 张工单创建成功。
- 高并发下瓶颈仍然指向 DB 连接池和同步写路径。

## 10. MCP 工具层

MCP server 当前暴露工具：

```text
ask_policy
create_ticket
continue_ticket_draft
confirm_action
ticket_tool_planner
get_ticket_detail
```

MCP health 小测：

```text
case=mcp_health
requests=5
concurrency=2
ok=5
error=0
p95≈232ms
```

MCP lookup 小测：

```text
case=mcp_lookup
requests=5
concurrency=2
ok=5
error=0
p95≈528ms
```

MCP lookup 大测：

```text
case=mcp_lookup
requests=1000
concurrency=100
ok=1000
error=0
rps≈27
p50≈3784ms
p95≈4258ms
p99≈4413ms
```

结论：

- MCP lookup 不经过 LLM，不是纯问答。
- MCP lookup 会走 MCP 协议、MCP wrapper、DB 读和 MCP 审计写入。
- 100 并发下没有错误，稳定性可以。
- 但 p95 超过 4 秒，说明延迟已经明显偏高。
- MCP 每次请求新建 session，包含 initialize、call_tool、关闭 session，协议开销比普通 HTTP 大。
- 和 API 直连 lookup 的 `1000/100` 对比，MCP lookup 反而没有大量超时，主要原因是 MCP 协议和 session 建立开销降低了实际打到 DB 的速度，相当于形成了排队/削峰；这不能说明 MCP 更快，只能说明它更慢但更不容易瞬间打爆 DB。

## 11. MCP create_ticket 说明

MCP `create_ticket` 不适合作为“纯 MCP 写路径上限”基准。

原因：

```text
MCP create_ticket
-> _handle_create_ticket_intent
-> run_ask_workflow
-> 字段抽取
-> 草稿判断
-> create_ticket_workflow
```

它可能混入 RAG/LLM/问答编排逻辑，因此不能和 FastAPI `/tickets` 直连建单压测直接对比。

如果要测试纯 MCP 写路径，建议后续暴露结构化、无 LLM 的 MCP 工具，例如：

```text
add_ticket_comment
escalate_ticket
create_ticket_direct
```

## 12. LLM API 压测位置

LLM API 压测应该作为最后的独立专项。

它不应该用于判断 MCP 工具层上限，因为 LLM API 会引入：

- 外部服务 rate limit
- token 生成耗时
- 网络波动
- 费用
- 模型队列延迟

推荐单独测试：

```text
LLM-only 并发
RAG 检索-only 并发
RAG + LLM 端到端并发
Agent planner + MCP 端到端并发
```

本轮压测没有继续打真实 LLM API，原因是本轮目标已经完成：

- 本地 API/DB 写路径瓶颈已经定位。
- MCP lookup 工具层表现已经定位。
- LLM API 属于外部服务限制，继续压测会引入费用、限流和网络波动，不适合混入本地系统瓶颈结论。

如果后续要查 LLM API 的并发限制，不能只靠本地压测猜，而应先查服务商限制。

以 OpenAI API 为例，重点查看：

```text
RPM: requests per minute，每分钟请求数
TPM: tokens per minute，每分钟 token 数
RPD: requests per day，每日请求数
TPD: tokens per day，每日 token 数
```

控制台位置：

```text
https://platform.openai.com/settings/organization/limits
```

实际 API 响应也可以观察 rate limit headers：

```text
x-ratelimit-limit-requests
x-ratelimit-remaining-requests
x-ratelimit-reset-requests
x-ratelimit-limit-tokens
x-ratelimit-remaining-tokens
x-ratelimit-reset-tokens
```

并发数通常不是一个固定配置项，而是由 RPM、TPM、平均 token 消耗和平均响应耗时共同决定。

估算方式：

```text
可持续 QPS = min(RPM / 60, TPM / 平均每请求 token数 / 60)
建议并发数 ≈ 可持续 QPS × 平均响应耗时秒数
```

例如：

```text
RPM = 300
TPM = 150000
平均每请求 3000 tokens
平均响应耗时 4 秒

请求数限制：300 / 60 = 5 requests/s
token限制：150000 / 3000 / 60 = 0.83 requests/s
建议并发：0.83 × 4 ≈ 3
```

因此，系统中 LLM 调用应该加客户端侧保护，例如：

- semaphore 控制同时在飞的 LLM 请求数。
- token bucket 控制每分钟请求数和 token 数。
- 对 429 做指数退避重试。
- LLM 压测单独进行，不和 DB/MCP/RAG 混压。

## 13. 总体结论

当前项目的瓶颈排序：

```text
入口层：不是主要瓶颈
DB 读路径：低中并发稳定，100 并发直连会超时
普通低风险写：低中并发稳定
热点更新写：50 稳定，60 开始雪崩
建单写路径：50 可通过，60 开始雪崩
MCP lookup：100 并发无错误，但 p95 约 4 秒
LLM API：未混入本轮结论，应单独测试
```

最核心瓶颈：

```text
同步数据库写路径在 50-60 并发附近触发 DB 连接池耗尽和同步线程池阻塞。
```

最关键证据：

```text
QueuePool limit of size 5 overflow 10 reached
```

## 14. 优化建议

优先级从高到低：

1. 对写操作加并发保护，超过阈值返回 429 或重试提示。
2. 对同一 `ticket_id` 的热点写操作做应用层削峰。
3. 审计日志异步化，减少主事务耗时。
4. 鉴权用户信息做短期缓存，减少每请求一次 DB 查用户。
5. 合理调整 SQLAlchemy 连接池大小，但不能盲目加大。
6. 后续再评估 SQLAlchemy AsyncSession / asyncpg 迁移。
7. MCP 高频调用考虑复用 session，减少每次 initialize 的协议开销。

## 15. 面试推荐表述

可以这样讲：

> 我没有只停留在功能实现上，而是做了分层并发测试。先绕过 MCP 直压 FastAPI，分别测入口、查单、评论、催办、建单，定位到底层 DB 写路径在 50-60 并发附近出现连接池耗尽和线程池阻塞；高并发查单也会因为鉴权和 DB session 排队出现超时。然后我再回到 MCP 层，单独压不经过 LLM 的 get_ticket_detail，确认 MCP 工具层 100 并发无错误，但 p95 到 4 秒左右，说明 MCP 协议和 session 建立有明显开销，同时也起到了削峰效果。整个过程中我没有把 LLM API 混进本地并发结论，因为 LLM 是外部依赖，应该单独测限流和延迟。最终我的判断是：系统一致性没问题，但缺少明确的背压机制，后续会优先做写操作限流、审计异步化、鉴权缓存和连接池调优。

## 16. 本轮压测状态

本轮压测到此结束。

已经形成的结论足够支撑面试叙事：

```text
1. 我做过分层压测，不是只凭感觉判断瓶颈。
2. 入口层不是最先瓶颈。
3. DB 写路径在 50-60 并发附近出现雪崩。
4. 读路径高并发也会因为鉴权和 DB session 排队超时。
5. MCP lookup 100 并发无错误，但延迟较高，协议/session 开销明显。
6. LLM API 没有混进本地压测结论，应按服务商 RPM/TPM 单独计算和限流。
```

后续如果继续优化，应该先做工程改进，而不是继续盲目加压。
