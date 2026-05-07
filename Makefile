.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: help
help:             	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:			## Create a virtual environment
	@echo "Creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "Run 'source .venv/bin/activate' to enable the environment"

.PHONY: install
install:		## Install dependencies
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -r requirements.txt

STRESS_URL = https://latam-delay-api-108332844354.us-central1.run.app
.PHONY: stress-test
stress-test:
	# change stress url to your deployed app
	mkdir reports || true
	locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H $(STRESS_URL)

.PHONY: model-test
model-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/model

.PHONY: api-test
api-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/api

.PHONY: build
build:			## Build locally the python artifact
	python setup.py bdist_wheel

# ----------------------------------------------------------------------------
# Part III — Docker + Cloud Run deploy targets
# ----------------------------------------------------------------------------

PROJECT_ID   := latam-challenge-495606
REGION       := us-central1
SERVICE_NAME := latam-delay-api
AR_REPO      := latam-images
IMAGE_TAG    := latest
IMAGE_REMOTE := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(AR_REPO)/$(SERVICE_NAME):$(IMAGE_TAG)

.PHONY: docker-build
docker-build:		## Build the Docker image locally
	docker build -t $(SERVICE_NAME):$(IMAGE_TAG) .

.PHONY: docker-auth
docker-auth:		## Configure docker as Artifact Registry credential helper
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet

.PHONY: docker-push
docker-push: docker-build docker-auth	## Build, tag, and push image to Artifact Registry
	docker tag $(SERVICE_NAME):$(IMAGE_TAG) $(IMAGE_REMOTE)
	docker push $(IMAGE_REMOTE)

.PHONY: deploy
deploy: docker-push	## Build + push image and update Cloud Run service to use it
	gcloud run services update $(SERVICE_NAME) \
		--image=$(IMAGE_REMOTE) \
		--region=$(REGION) \
		--project=$(PROJECT_ID)

# ----------------------------------------------------------------------------
# Terraform shortcuts (operate inside infra/)
# ----------------------------------------------------------------------------

.PHONY: tf-init
tf-init:		## Initialize Terraform inside infra/
	cd infra && terraform init

.PHONY: tf-plan
tf-plan:		## Preview Terraform changes
	cd infra && terraform plan

.PHONY: tf-apply
tf-apply:		## Apply Terraform changes (creates GCP infra)
	cd infra && terraform apply

.PHONY: tf-destroy
tf-destroy:		## Destroy all Terraform-managed GCP infra (cleanup)
	cd infra && terraform destroy

.PHONY: tf-output
tf-output:		## Show Terraform outputs (Cloud Run URL, AR URL, etc.)
	cd infra && terraform output
