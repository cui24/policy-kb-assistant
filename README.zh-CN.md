# Policy KB Assistant

[English](README.en.md) | [简体中文](README.md)

企业知识库问答 + ITSM 智能体。这个项目把政策问答、工单工作流和通过 MCP 暴露的工具统一放在同一套受约束的执行层之下。

核心设计很明确：
- 模型只负责提出路由或工具计划
- 后端负责校验 schema、鉴权和确认态
- 只有通过校验的动作才会执行，并且全链路可审计

两个关键差异点：
- 同一套 skills registry 同时驱动 `/agent` 和 MCP tools
- 高风险动作采用两步确认，而不是直接执行

## 前置依赖

- Python 3.10+
- 推荐使用 Conda 或 Miniconda 维护本地环境
- 推荐使用 `make`，因为仓库已提供 `Makefile`
- 只有完整版演示才需要 Docker + Docker Compose

## 可演示能力

- 做带 citations 的政策问答（网页自然语言入口统一走 `POST /agent`，`POST /ask` 仍保留为直接 API 入口）
- 通过 `POST /agent` 做一句话路由
- 工单创建、查询、追加评论、催办、取消
- 调用真实 HTTP API 的 Streamlit 网页
- 暴露受约束 ticket tools 的 MCP stdio server
- 通过 `kb_queries` 和 `audit_logs` 回放链路

## Quickstart（最小本地运行）

这条路径适合最快跑通工单、网页和 MCP tools。不需要 Docker、Postgres 或 Qdrant。

1. 安装依赖。

```bash
conda create -n policy-kb python=3.10 -y
conda activate policy-kb
python -m pip install -r requirements.txt
cp .env.example .env
```

2. 在 `.env` 中覆盖为 SQLite 本地运行。

```dotenv
DATABASE_URL=sqlite:///./policy_kb_l2.db
POLICY_API_KEY=local-dev-key
AUTO_MIGRATE_ON_STARTUP=true
DEV_DB_FALLBACK_CREATE_ALL=false
```

3. 启动 API。

```bash
make api
```

4. 在另一个终端启动网页。

```bash
make ui
```

5. 打开页面。

- 网页：`http://localhost:8501`
- API 健康检查：`http://localhost:8080/health`

网页侧边栏建议填：
- `API Base URL`: `http://localhost:8080`
- `API Key`: `local-dev-key`
- `User`: `alice`
- `Department`: `IT`

这个模式下可用：
- 手动创建工单
- 查询和管理工单
- 回查已创建工单的追溯信息
- MCP ticket tools

这个模式下的限制：
- 知识库问答（`POST /ask` 或被 `/agent` 路由为 `ASK`）需要有效的 `OPENAI_API_KEY`
- 检索式问答还需要 Qdrant 和已入库文档

## Docker Compose（部署版，一键可用）

这条路径适合“交付演示”或“评审环境”：一次拉起 API + UI + Postgres + Redis + Qdrant，并自动做 KB 初始化。

1. 准备 `.env`（保持 Postgres 风格配置），至少设置：

```dotenv
POLICY_API_KEY=local-dev-key
OPENAI_API_KEY=YOUR_REAL_OPENAI_COMPATIBLE_KEY
```

2. 一键启动并构建镜像：

```bash
docker compose up -d --build
# 或
make up-build
```

3. 检查容器与 KB 初始化结果：

```bash
docker compose ps -a
docker compose logs --tail=100 kb-init
```

4. 打开页面：

- 网页：`http://localhost:8501`
- API 文档：`http://localhost:8080/docs`
- API 健康检查：`http://localhost:8080/health`
- Qdrant Dashboard：`http://localhost:6333/dashboard`

网页打开后先注册或登录一个 demo 用户。UI 容器会通过 `POLICY_API_BASE_URL=http://api:8080`
访问后端 API，本机浏览器只需要打开 `http://localhost:8501`。

这个模式下可以演示：
- 网页经 `/agent` 路由的知识库问答
- 如果你直接调 API，也可以单独调用 `POST /ask`
- 草稿续办
- 既有工单工具动作
- Web 和 MCP 双入口

说明：
- 首次启动可能较慢（需要模型下载和首次入库）。
- 后续重启通常会更快，`kb-init` 会在集合已就绪时跳过入库。

文档来源说明和可选外链见 [docs/demo_data.md](docs/demo_data.md)。

## Docker Compose（开发版，挂载代码 + 热更新）

这条路径适合日常开发：代码改动可实时生效，不用每次重建镜像。

1. 启动开发环境：

```bash
make dev-up
```

2. 查看 API/UI 日志：

```bash
docker compose -f compose.yaml -f compose.dev.yaml logs -f api ui
```

3. 停止开发环境：

```bash
make dev-down
```

开发版特性：
- `api` 使用 `uvicorn --reload`，修改 `src/api/` 会自动重载。
- `ui` 使用文件轮询，修改 `src/ui/` 会自动刷新。
- 通过 bind mount 挂载代码，容器内直接使用你本地最新文件。

开发版注意：
- 改了 `requirements.txt` 或 `Dockerfile.*` 后，仍需重新构建（执行 `make dev-up`）。
- UI 在容器里访问 API 时，必须使用 `http://api:8080`，不要填 `http://localhost:8080`。

## MCP

默认推荐远程 MCP（业务场景）：
- `AGENT_MCP_CLIENT_ENABLED=true`
- `AGENT_MCP_SERVER_URL=http://mcp:9000/mcp`（容器内）或 `http://127.0.0.1:9000/mcp`（本机联调）

本地 `stdio` 仅用于开发调试与 smoke。

以 demo 模式启动 stdio server：

```bash
export MCP_ACTOR_USER_ID=alice
export MCP_DEPARTMENT=IT
PYTHONPATH=$(pwd) python -m src.mcp_stdio_server
```

如果你不想接外部 Host，也可以直接跑本地 smoke：

```bash
PYTHONPATH=$(pwd) python scripts/mcp_smoke.py --actor alice
```

## 安全模型

这个仓库实现的是 demo 级安全模型，不是生产级多租户鉴权系统。

- `/agent` 和写接口使用共享 `X-API-Key`
- MCP stdio 通过 `MCP_ACTOR_USER_ID` 绑定固定 actor
- 高风险取消必须两步：先申请确认，再执行确认
- 审计记录会在 payload 中标记来源，便于区分 Web 和 MCP

当前明确不做：
- OAuth
- 按用户映射的 bearer token
- 支持多用户身份映射的远程 HTTP MCP

## Makefile 的作用

`Makefile` 是项目的统一命令入口，用来把“长命令”和“易错参数”收敛成短命令，降低操作成本。

你可以把它理解为：
- 给团队约定一套标准动作（启动、停止、测试、入库、迁移）。
- 把 Docker/Conda/PYTHONPATH 等细节藏在命令背后。
- 降低手敲命令时漏参数、输错文件名的风险。

当前常用目标：
- `make up`：启动部署版容器（不强制重建镜像）
- `make up-build`：启动部署版容器并重建镜像
- `make down`：停止部署版容器
- `make dev-up`：启动开发版（挂载代码 + 热更新）
- `make dev-restart`：重启开发版 `api` 和 `ui`
- `make dev-down`：停止开发版容器
- `make api` / `make ui`：本机 Conda 环境运行 API/UI（非容器）
- `make ingest`：执行文档入库
- `make test`：运行测试

## 测试

运行本地回归测试：

```bash
PYTHONPATH=$(pwd) pytest -q tests
```

仓库当前包含：
- services 层测试
- API smoke tests
- MCP in-memory tool tests
- Streamlit UI smoke tests

GitHub Actions 会在 push 和 pull request 时跑核心测试子集。

## 常见问题

- `401 Unauthorized`：请先在网页登录/注册；直接调用受保护 API 时需要携带 `Authorization: Bearer <token>`。
- `/agent` 报 500 且日志提示 `collection ... doesn't exist`：先看 `kb-init` 日志；如初始化失败，执行 `docker compose logs kb-init` 排查，再手动执行一次 `docker compose exec -T api python -m src.kb.ingest`。
- 没有 citations 或 KB 返回为空：通常是没执行 `make ingest`，或没有配置有效的 `OPENAI_API_KEY`。
- MCP host 无法连接：stdio 模式不要向 `stdout` 打日志，请改用 `stderr`。
- FastAPI `on_event` deprecation warning：这是已知 warning，不影响运行，后续会迁移到 `lifespan`。

## 仓库结构

- `src/api/`: FastAPI、services、skills registry、持久化逻辑
- `src/ui/`: Streamlit 页面和 HTTP 客户端
- `src/kb/`: 检索和答案生成
- `src/agent/`: 工单字段抽取
- `src/mcp_stdio_server.py`: MCP stdio 入口
- `tests/`: 回归测试
- `scripts/`: smoke 和发布辅助脚本
- `docs/`: 对外文档

## 文档

- [Architecture](docs/architecture.md)
- [Demo Data](docs/demo_data.md)
- [MCP](docs/mcp.md)
