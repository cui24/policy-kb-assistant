# 默认 Conda 环境名；可在命令行覆盖，例如：`make test CONDA_ENV=myenv`
CONDA_ENV ?= policy-kb
# 统一 Python 运行入口：确保所有 Python 脚本都在同一个 Conda 环境执行
PYTHON_RUN = conda run -n $(CONDA_ENV) python
# 统一 Alembic 运行入口：额外注入 PYTHONPATH，保证能找到项目内的 `src.*` 包
ALEMBIC_RUN = PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) alembic

# 声明伪目标：避免目录/同名文件与目标名冲突导致命令不执行
.PHONY: up up-build dev-up dev-restart down dev-down download check-pdf smoke ingest retrieve demo regress test ui api mcp-install mcp-stdio mcp-http mcp-smoke compose-build-mcp db-upgrade db-current db-history db-revision grid-search ci-local ci-test ci-eval

# 部署版启动（不强制重建镜像）：已有镜像时更快，适合日常重启服务
up:
	# 使用默认 `compose.yaml` 后台启动全部服务
	docker compose up -d

# 部署版启动并重建镜像：代码或依赖变更后使用
up-build:
	# 强制在启动前执行 build，确保容器使用最新镜像
	docker compose up -d --build

# 开发版启动（挂载代码 + 热更新）并重建镜像
dev-up:
	# 叠加 `compose.dev.yaml`，启用 bind mount 与 reload 配置
	docker compose -f compose.yaml -f compose.dev.yaml up -d --build

# 开发版只重启 API 和 UI：用于改了环境变量或需要快速重启进程的场景
dev-restart:
	# 仅重启 `api` 与 `ui`，数据库和中间件不中断
	docker compose -f compose.yaml -f compose.dev.yaml restart api ui

# 部署版停止并删除容器/网络
down:
	# 释放默认 compose 项目的运行资源
	docker compose down

# 开发版停止并删除容器/网络
dev-down:
	# 按开发版叠加配置停止，避免残留 dev 容器
	docker compose -f compose.yaml -f compose.dev.yaml down

# 下载额外 PDF 示例数据到本地 `data/raw/`
download:
	# 执行仓库内下载脚本（`data/raw/` 默认 gitignore）
	./scripts/download_pdfs.sh

# 检查 demo PDF 是否能正常抽取文本
check-pdf:
	# 对 `data/demo` 目录做 PDF 可读性检查
	$(PYTHON_RUN) scripts/check_pdf_text.py data/demo

# 最小占位 smoke 测试（环境连通性快速检查）
smoke:
	# 运行占位模块，验证 Python 入口与包导入是否正常
	$(PYTHON_RUN) -m src.kb.placeholder

# 将文档向量化并写入 Qdrant
ingest:
	# 执行知识库入库流程（首次会下载模型，耗时较长）
	$(PYTHON_RUN) -m src.kb.ingest

# 执行一次检索，参数通过 `Q` 传入，例如：`make retrieve Q="校园网报修"`
retrieve:
	# 调用检索模块，输出 top-k 证据
	$(PYTHON_RUN) -m src.kb.retrieve "$(Q)"

# 运行命令行问答 demo，参数通过 `Q` 传入
demo:
	# 调用 CLI 演示入口，便于无 UI 快速验证
	$(PYTHON_RUN) -m src.cli.demo_cli "$(Q)"

# 执行回归评测脚本
regress:
	# 运行评测模块，检查问答质量和行为回归
	$(PYTHON_RUN) -m src.eval.run_regression

# 运行测试集
test:
	# 注入项目根路径后执行 pytest，覆盖 `tests/` 下全部测试
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) pytest tests

# 本机模式启动 Streamlit UI（非容器）
ui:
	# 直接在 Conda 环境运行前端页面，默认端口 8501
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) streamlit run src/ui/app.py --server.headless true --browser.gatherUsageStats false

# 本机模式启动 FastAPI（非容器）
api:
	# 直接在 Conda 环境运行 API，监听 0.0.0.0:8080
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) uvicorn src.api.app:app --host 0.0.0.0 --port 8080

# 在本机 Conda 环境安装 MCP 依赖（与主 requirements 解耦）
mcp-install:
	# 安装 `requirements-mcp.txt`，用于 stdio MCP server/测试
	conda run -n $(CONDA_ENV) python -m pip install -r requirements-mcp.txt

# 本机模式启动 MCP stdio server（供 Host/IDE 通过 stdio 接入）
mcp-stdio:
	# 需在 shell 中先设置 `MCP_ACTOR_USER_ID`；可选 `MCP_DEPARTMENT`
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) python -m src.mcp_stdio_server

# 本机模式启动 MCP HTTP server（供远端客户端通过 HTTP 接入）
mcp-http:
	# 需在 shell 中先设置 `MCP_ACTOR_USER_ID`；可选 `MCP_HOST/MCP_PORT`
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) python -m src.mcp_http_server

# 运行 MCP in-process smoke 测试（不依赖外部 Host）
mcp-smoke:
	# 可传参：`ACTOR=alice DEPARTMENT=IT`
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) python scripts/mcp_smoke.py --actor "$(or $(ACTOR),mcp-demo-user)" --department "$(or $(DEPARTMENT),IT)"

# 构建带 MCP 依赖的 API/KB 初始化镜像（可选）
compose-build-mcp:
	# 通过 build-arg 打开 MCP 依赖安装
	docker compose build --build-arg INSTALL_MCP=1 api kb-init mcp

# 数据库迁移到最新版本
db-upgrade:
	# 执行 Alembic upgrade head
	$(ALEMBIC_RUN) upgrade head

# 查看当前数据库版本
db-current:
	# 输出当前 DB 对应的 migration revision
	$(ALEMBIC_RUN) current

# 查看迁移历史
db-history:
	# 列出 Alembic 历史 revision 链
	$(ALEMBIC_RUN) history

# 基于模型变更自动生成迁移文件，消息通过 `MSG` 传入
db-revision:
	# 示例：`make db-revision MSG="add_user_table"`
	$(ALEMBIC_RUN) revision --autogenerate -m "$(MSG)"

# 执行检索门控参数网格搜索
grid-search:
	# 运行参数搜索模块，产出候选配置与评估结果
	$(PYTHON_RUN) -m src.eval.grid_search_gate

# 容器内 CI：确定性测试 + 轻量质量评测
ci-local: ci-test ci-eval

# 容器内确定性测试：覆盖语法、记忆、Agent Graph、MCP wrapper 和关键服务回归
ci-test:
	docker compose -f compose.yaml -f compose.dev.yaml exec -T api \
		env PYTHONPATH=/app python -m compileall -q src tests scripts
	docker compose -f compose.yaml -f compose.dev.yaml exec -T api \
		env PYTHONPATH=/app python -m pytest -q \
			tests/test_memory_l1_session.py \
			tests/test_agent_graph_working_memory.py \
			tests/test_agent_graph_mcp_client.py \
			tests/test_mcp_wrapper_executor.py \
			tests/test_services.py -k "memory or recent_draft or recent_ticket"

# 容器内轻量质量门禁：固定 RAG fixture 指标 + Agent workflow 小样本评测
ci-eval:
	docker compose -f compose.yaml -f compose.dev.yaml exec -T api \
		env PYTHONPATH=/app python scripts/ci_quality_gate.py --out-dir outputs/ci
