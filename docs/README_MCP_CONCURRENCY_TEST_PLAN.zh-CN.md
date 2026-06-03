# MCP / 后端并发测试方案

本文档用于记录 `policy-kb-assistant` 的并发测试思路。目标不是证明系统能承受多高并发，而是分层发现短板：入口层、MCP wrapper、Redis、数据库、LLM/RAG、审计日志到底哪里先成为瓶颈。

## 1. 测试目标

并发测试重点观察：

- 入口层是否稳定接收请求。
- Redis 限流、幂等、确认态是否成为瓶颈。
- 数据库连接池、写入、行锁是否成为瓶颈。
- 同一工单并发更新是否出现状态覆盖。
- 审计日志写入是否造成写放大。
- LLM / RAG 链路是否触发超时或 API 限流。

## 2. 推荐启动方式

并发测试不要使用 SQLite 本地最小模式。SQLite 写并发能力弱，会把问题放大到不真实。

推荐使用 Docker Compose，至少包含：

```text
api
mcp
postgres
redis
qdrant
```

启动：

```bash
docker compose up -d --build
```

或开发模式：

```bash
make dev-up
```

检查服务：

```bash
docker compose ps
curl http://localhost:8080/health
```

如需测试 MCP HTTP，确认 MCP server 地址，例如：

```text
http://localhost:9000/mcp
```

UI 不需要参与并发测试。

## 3. 测试前准备

可以使用准备脚本自动完成注册/登录与测试工单创建：

```bash
python scripts/concurrency_smoke.py
```

脚本当前会执行：

- 读取 API 地址、测试用户名、密码、部门等配置。
- 调用 `/auth/register`，如果用户已存在则跳过。
- 调用 `/auth/login` 获取 Bearer Token。
- 创建四张测试工单：
  - `lookup`：用于查单。
  - `comment`：用于追加评论。
  - `escalate`：用于催办。
  - `cancel`：用于取消确认。
- 将结果写入：

```text
outputs/concurrency_setup.json
```

可选配置：

```bash
LOADTEST_API_BASE_URL=http://localhost:8080 \
LOADTEST_USERNAME=loadtest_user \
LOADTEST_PASSWORD=loadtest_password_123 \
LOADTEST_DEPARTMENT=IT \
python scripts/concurrency_smoke.py
```

如需每次创建全新的测试工单：

```bash
python scripts/concurrency_smoke.py --fresh
```

当前脚本已支持六个最小并发 case：

```bash
# 入口层基线
python scripts/concurrency_smoke.py --case health --requests 100 --concurrency 10

# 带鉴权的查单读并发
python scripts/concurrency_smoke.py --case lookup --requests 100 --concurrency 10

# 同一工单追加评论写并发
python scripts/concurrency_smoke.py --case comment --requests 100 --concurrency 10

# 同一工单催办更新并发，并检查 escalation_count 是否丢失
python scripts/concurrency_smoke.py --case escalate --requests 100 --concurrency 10

# 多工单分散催办更新并发，默认准备 20 张工单
python scripts/concurrency_smoke.py --case escalate_many --requests 600 --concurrency 60 --distributed-tickets 20

# 并发创建工单，覆盖 Redis 幂等检查和 DB 插入写路径
python scripts/concurrency_smoke.py --case create --requests 100 --concurrency 10
```

MCP 工具层并发测试使用独立脚本：

```bash
# MCP list_tools 基线
python scripts/mcp_concurrency_smoke.py --case health --requests 20 --concurrency 5

# MCP get_ticket_detail 并发；未传 ticket-id 时会自动准备一张 demo_user 的种子工单
python scripts/mcp_concurrency_smoke.py --case lookup --requests 50 --concurrency 5

# MCP create_ticket 并发
python scripts/mcp_concurrency_smoke.py --case create --requests 50 --concurrency 5
```

注意：MCP 脚本需要当前 Python 环境安装 `mcp` 包；如果宿主机环境缺少依赖，可以在安装项目依赖的环境中运行。

注意：输出文件中包含本地测试用 `access_token`，不要提交或分享。

如果测试 RAG/问答，另需准备：

  - 确认 Qdrant 已入库。
  - 配置有效 `OPENAI_API_KEY`。
  - 先预热一次问答，避免冷启动影响结果。

## 4. 分层测试顺序

不要一开始就做混合压测。建议按下面顺序逐层定位瓶颈。

### 4.1 入口层基线

目标：确认 FastAPI / MCP HTTP 入口是否稳定。

测试接口：

```text
GET /health
```

并发阶梯：

```text
1 -> 10 -> 50 -> 100 -> 200
```

观察：

- p50 / p95 / p99 latency。
- error rate。
- API 容器 CPU / memory。

判断：

- 如果 `/health` 都慢，优先检查 worker、容器资源和入口配置。
- 如果 `/health` 稳定，入口层通常不是最主要瓶颈。

### 4.2 工单只读并发

目标：测试数据库读压力。

测试接口：

```text
GET /tickets/{ticket_id}
```

或 MCP tool：

```text
get_ticket_detail
```

测试方式：

- 多客户端反复查询同一张工单。
- 多客户端查询不同工单。

观察：

- DB 连接池是否耗尽。
- Postgres CPU。
- API p95 latency。
- 是否出现 5xx。

### 4.3 建单写入并发

目标：测试创建工单、Redis 幂等、审计写入压力。

测试接口：

```text
POST /tickets
```

或 MCP tool：

```text
create_ticket
```

注意：

- 每个请求使用不同 `Idempotency-Key`，测试真实写入吞吐。
- 再单独用相同 `Idempotency-Key`，测试 replay / conflict 行为。

观察：

- Redis 幂等耗时。
- Postgres insert 速度。
- audit_logs 写入压力。
- 是否产生重复工单。

### 4.4 同一工单并发评论

目标：测试 append-only 评论和审计写入。

测试接口：

```text
POST /tickets/{ticket_id}/comments
```

或 MCP tool：

```text
add_ticket_comment
```

测试方式：

```text
50 个并发同时给同一张工单追加评论
```

观察：

- 评论数量是否正确。
- 是否丢评论。
- `ticket.updated_at` 是否正常。
- audit_logs 是否完整写入。
- p95 latency 是否升高。

### 4.5 同一工单状态冲突

目标：测试行锁、状态冲突和事务行为。

测试场景：

```text
多个客户端同时催办同一张工单
催办和取消同时发生
取消确认和评论同时发生
```

观察：

- 最终状态是否合理。
- 是否出现 cancelled 后又变回 in_progress。
- 是否出现 500。
- 是否出现锁等待过长。

这组测试重点不是吞吐，而是一致性。

### 4.6 LLM / RAG 并发

目标：测试外部模型、Qdrant、embedding / rerank 是否成为瓶颈。

测试接口：

```text
POST /ask
POST /agent
```

并发阶梯建议较低：

```text
1 -> 3 -> 5 -> 10 -> 20
```

观察：

- LLM API 是否限流。
- Qdrant latency。
- answer latency。
- 是否出现 timeout。

这组链路通常最重，不要和普通工单工具混在一起做第一轮压测。

## 5. 压测工具选择

### 5.1 hey / wrk

适合测试普通 HTTP API：

```bash
hey -n 1000 -c 50 http://localhost:8080/health
```

MCP streamable HTTP 不一定适合直接用 `hey`，更适合用 MCP SDK client。

### 5.2 Python + httpx asyncio

适合自定义：

- 并发数。
- 请求体。
- Bearer Token。
- Idempotency-Key。
- p95 / p99 统计。
- 错误码分布。

这是当前项目最适合的压测脚本方向。

### 5.3 Locust

适合模拟多用户行为：

```text
登录 -> 建单 -> 查单 -> 评论 -> 等待 -> 再查单
```

学习成本更高，可作为未来方案。

## 6. 需要记录的指标

每组测试记录：

- 并发数。
- 总请求数。
- 成功数。
- 失败数。
- 错误码分布。
- 平均延迟。
- p50 / p95 / p99。
- 最大延迟。
- 吞吐 req/s。

服务侧观察：

```bash
docker compose logs -f api
docker compose logs -f mcp
docker compose logs -f postgres
docker compose logs -f redis
docker compose logs -f qdrant
```

重点关注：

```text
DB connection pool timeout
Redis unavailable
rate_limited
idempotency_conflict
LLM timeout
Qdrant timeout
deadlock
lock timeout
500 internal_error
```

## 7. 如何判断瓶颈

| 现象 | 可能瓶颈 |
| --- | --- |
| `/health` 慢 | 入口 worker / 容器资源 |
| 查单慢 | DB 读 / 连接池 |
| 建单慢 | DB 写 / Redis 幂等 / audit log |
| 评论慢 | DB 写放大 / audit log |
| 同一工单状态异常 | 并发一致性问题 |
| `/ask` 慢或失败 | LLM / Qdrant / embedding / rerank |
| 大量 `rate_limited` | 并发闸门或限流策略触发 |

## 8. 最后再做混合场景

单项测试完成后，再做接近真实 MCP 使用的混合压测，例如：

```text
70% 查单
15% 建单
10% 评论
5% 催办/取消
```

混合压测可以暴露资源竞争，例如 DB 写入、Redis 幂等和审计日志同时被打满。

## 9. 面试表达版本

可以这样回答：

> 我如果要验证 MCP 并发能力，不会一上来做混合压测，而是先分层压。先打 `/health` 看入口，再打查单看 DB 读，再打建单看 Redis 幂等和 DB 写，再打评论和催办看同一工单并发一致性，最后单独测 RAG/LLM，因为模型链路和普通工单工具瓶颈完全不同。指标上主要看 p95/p99、错误率、DB 连接池、Redis 错误、LLM 限流和审计写入压力。这样能定位到底是入口层、DB、Redis、LLM 还是审计日志先成为短板。
