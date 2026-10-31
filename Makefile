.PHONY: help test fixture run image-gpu clean validate

SERVICE_NAME ?= sc-integrate
PYTHON ?= python3
CFFCONVERT ?= pipx run cffconvert==2.0.0
export DOCKER_DEFAULT_PLATFORM = linux/amd64

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test      - pytest: the synthetic fixture through the processor, Harmony and scVI, both input layouts (no Docker)"
	@echo "make fixture   - write the synthetic sc-qc output to data/fixture/{linked,flat}/"
	@echo "make run       - build the cpu image and run it in Docker over data/fixture/linked/, writing to data/output/{cells,labels,integration}/"
	@echo "make image-gpu - build the gpu variant's image (CUDA PyTorch) as $(SERVICE_NAME):gpu"
	@echo "make validate  - app.yml against the manifest schema, CITATION.cff with cffconvert"
	@echo "make clean     - remove data/output/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"
	@echo "CFFCONVERT=$(CFFCONVERT)"

test:
	MPLBACKEND=Agg $(PYTHON) -m pytest -q tests

fixture:
	rm -rf data/fixture
	$(PYTHON) -m tests.make_fixture data/fixture --layout both

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

image-gpu:
	docker build --platform linux/amd64 --build-arg TORCH_FLAVOUR=cuda -t $(SERVICE_NAME):gpu .

validate:
	$(PYTHON) -c "import json, yaml, jsonschema; s = json.load(open('../workflow-single-cell/docs/app-manifest.v1.schema.json')); jsonschema.Draft202012Validator(s).validate(yaml.safe_load(open('app.yml'))); print('app.yml ok')"
	$(CFFCONVERT) --validate && echo "CITATION.cff ok"

clean:
	rm -rf data/output
