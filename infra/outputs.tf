output "cloud_run_url" {
  description = "Public HTTPS URL of the Cloud Run service."
  value       = google_cloud_run_v2_service.api.uri
}

output "ar_repo_url" {
  description = "Docker repository URL (use this as the registry for `docker tag`/`docker push`)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ar_repo_id}"
}

output "image_url" {
  description = "Full image reference for the API. Push to this URL and Cloud Run will pull from it."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.ar_repo_id}/${var.service_name}:latest"
}

output "deployer_service_account" {
  description = "Service account email used by GitHub Actions to deploy (Part IV)."
  value       = google_service_account.deployer.email
}

output "monitoring_dashboard_url" {
  description = "Direct link to the Cloud Monitoring dashboard."
  value       = "https://console.cloud.google.com/monitoring/dashboards/builder/${google_monitoring_dashboard.api.id}?project=${var.project_id}"
}
