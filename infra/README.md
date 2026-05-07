# Infra — Terraform

Provisions the GCP infrastructure for the LATAM Flight Delay Prediction API:

- **Artifact Registry** Docker repo (`latam-images`)
- **Cloud Run** service (`latam-delay-api`) — initially with a placeholder image
- **IAM** service account (`latam-deployer`) for the Part IV CD pipeline
- **Cloud Monitoring** custom dashboard (RPS, latency p95, 5xx rate, instances)

## Prerequisites

- `gcloud` CLI authenticated:
  ```bash
  gcloud auth login
  gcloud auth application-default login
  gcloud config set project latam-challenge
  ```
- The following GCP APIs enabled in the project:
  - `run.googleapis.com`
  - `artifactregistry.googleapis.com`
  - `iam.googleapis.com`
  - `cloudresourcemanager.googleapis.com`
  - `monitoring.googleapis.com`
- Billing enabled on the project.

## Apply

```bash
cd infra
terraform init
terraform plan
terraform apply
```

After apply, get the outputs:

```bash
terraform output
# cloud_run_url            = "https://latam-delay-api-xxxxx.run.app"
# ar_repo_url              = "us-central1-docker.pkg.dev/latam-challenge/latam-images"
# image_url                = "us-central1-docker.pkg.dev/latam-challenge/latam-images/latam-delay-api:latest"
# deployer_service_account = "latam-deployer@latam-challenge.iam.gserviceaccount.com"
# monitoring_dashboard_url = "https://console.cloud.google.com/monitoring/..."
```

## Image lifecycle

Terraform creates the Cloud Run service with a placeholder image
(`us-docker.pkg.dev/cloudrun/container/hello`). The real image is built and
pushed outside Terraform — by `make deploy` locally, or by the CD pipeline
(Part IV).

`template[0].containers[0].image` is in `lifecycle.ignore_changes`, so
post-deploy image swaps are NOT treated as Terraform drift.

## Destroy

```bash
terraform destroy
```

This removes the Cloud Run service, AR repo (and any images in it), the
deployer SA, IAM bindings, and the monitoring dashboard. Useful at the end of
the challenge to leave the GCP project clean.

## State

State is stored locally in `terraform.tfstate` and gitignored. For a
multi-developer setup we would use a GCS backend; for this challenge local
state is sufficient.
