variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-south-2"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "tfm-slm"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g6.2xlarge"
}

variable "ssh_key_name" {
  description = "Name of the SSH key pair"
  type        = string
  default     = "tfm-slm"
}

variable "docker_image_tag" {
  description = "Tag of the Docker image to deploy"
  type        = string
  default     = "latest"
}

variable "spot_price" {
  description = "Maximum price for Spot instance"
  type        = string
  default     = "0.50"
}
