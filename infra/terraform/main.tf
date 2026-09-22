terraform {
  required_version = ">= 1.8.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.23.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type    = string
  default = "us-central1-a"
}

variable "run_id" {
  type = string
}

variable "machine_type" {
  type    = string
  default = "g2-standard-8"
}

variable "boot_disk_gb" {
  type    = number
  default = 200
}

variable "image" {
  type    = string
  default = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
}

locals {
  labels = {
    project = "mini-ai-stack"
    owner   = "project-owner"
    run_id  = var.run_id
  }
}

resource "google_compute_instance" "campaign" {
  name         = "mini-ai-stack-${var.run_id}"
  machine_type = var.machine_type
  labels       = local.labels

  boot_disk {
    initialize_params {
      image = var.image
      size  = var.boot_disk_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
  }

  # g2-standard-8 includes one NVIDIA L4; do not add a separate accelerator block.
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  metadata = {
    enable-oslogin = "TRUE"
  }
}
