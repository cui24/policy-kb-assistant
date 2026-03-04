# Policy KB Assistant

[English](README.md) | [简体中文](README.zh-CN.md)

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

## Full Demo（Postgres + Qdrant + 文档入库）

这条路径适合演示完整的问答 + 工单 + 审计链路。

1. 启动基础设施。

```bash
make l2-up
```

会拉起：
- Qdrant
- Postgres
- Redis

2. 保持 `.env.example` 里默认的 Postgres 风格 `DATABASE_URL`，并设置：

```dotenv
POLICY_API_KEY=local-dev-key
OPENAI_API_KEY=YOUR_REAL_OPENAI_COMPATIBLE_KEY
```

3. 仓库默认自带一个可公开再分发的示例文档：

- `data/demo/ACME_IT_Admin_Handbook_v1.0_Demo.pdf`

其他来源文档不随仓库分发。如果你想用开发时的额外 PDF，可以下载到 `data/raw/`：

```bash
./scripts/download_pdfs.sh
```

`data/raw/` 默认被 `.gitignore` 忽略，所以这些额外文档只保留在本地。

4. 将自带 demo 文档和本地 `data/raw/` 文档一起入库到 Qdrant。

```bash
make ingest
```

5. 启动 API 和网页。

```bash
make api
make ui
```

这个模式下可以演示：
- 网页经 `/agent` 路由的知识库问答
- 如果你直接调 API，也可以单独调用 `POST /ask`
- 草稿续办
- 既有工单工具动作
- Web 和 MCP 双入口

文档来源说明和可选外链见 [docs/demo_data.md](docs/demo_data.md)。

## MCP

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

- `401 Unauthorized`：请在 `.env` 里设置 `POLICY_API_KEY`，并在网页侧边栏填同样的值。
- `Qdrant connection refused`：完整版演示需要先执行 `make l2-up`，再执行 `make ingest` 或跑依赖知识库的 `/agent`。
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
