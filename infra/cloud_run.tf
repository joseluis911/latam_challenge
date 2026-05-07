locals {
  # Placeholder image used by Terraform on first apply. The real image is
  # pushed by the CD pipeline (or `make deploy` locally) and applied with
  # `gcloud run services update --image <ar-image>:latest`. The lifecycle
  # block below tells Terraform not to treat post-deploy image changes as drift.
  placeholder_image = "us-docker.pkg.dev/cloudrun/container/hello"
}

resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region

  template {
    containers {
      image = local.placeholder_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
      # Note: Cloud Run sets PORT automatically based on ports.container_port.
      # Setting it manually triggers "reserved env names" error.
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    timeout = "60s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }

  depends_on = [google_artifact_registry_repository.api]
}

# Public unauthenticated invocation (the LATAM evaluator POSTs without auth).
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = google_cloud_run_v2_service.api.project
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
