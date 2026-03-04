CONDA_ENV ?= policy-kb
PYTHON_RUN = conda run -n $(CONDA_ENV) python
ALEMBIC_RUN = PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) alembic

l0-up:
	docker compose -f infra/docker-compose.yml --profile l0 up -d

l2-up:
	docker compose -f infra/docker-compose.yml --profile l2 up -d

l4-up:
	docker compose -f infra/docker-compose.yml --profile l4 up -d

l3-up:
	docker compose -f infra/docker-compose.yml --profile l2 up -d
	docker compose -f infra/docker-compose.gpu.yml --profile l3 up -d

down:
	docker compose -f infra/docker-compose.yml down
	docker compose -f infra/docker-compose.gpu.yml down || true

download:
	./scripts/download_pdfs.sh

check-pdf:
	$(PYTHON_RUN) scripts/check_pdf_text.py data/demo

smoke:
	$(PYTHON_RUN) -m src.kb.placeholder

ingest:
	$(PYTHON_RUN) -m src.kb.ingest

retrieve:
	$(PYTHON_RUN) -m src.kb.retrieve "$(Q)"

demo:
	$(PYTHON_RUN) -m src.cli.demo_cli "$(Q)"

regress:
	$(PYTHON_RUN) -m src.eval.run_regression

test:
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) pytest tests

ui:
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) streamlit run src/ui/app.py --server.headless true --browser.gatherUsageStats false

api:
	PYTHONPATH=$(CURDIR) conda run -n $(CONDA_ENV) uvicorn src.api.app:app --host 0.0.0.0 --port 8080

db-upgrade:
	$(ALEMBIC_RUN) upgrade head

db-current:
	$(ALEMBIC_RUN) current

db-history:
	$(ALEMBIC_RUN) history

db-revision:
	$(ALEMBIC_RUN) revision --autogenerate -m "$(MSG)"

grid-search:
	$(PYTHON_RUN) -m src.eval.grid_search_gate
