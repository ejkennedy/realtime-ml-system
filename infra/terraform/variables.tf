variable "project_name" {
  default = "fraud-detection"
}

variable "environment" {
  default = "prod"
  validation {
    condition     = contains(["prod", "staging", "dev"], var.environment)
    error_message = "Environment must be prod, staging, or dev."
  }
}

variable "aws_region" {
  default = "us-east-1"
}
