terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "tfm-slm-terraform-state"
    key            = "terraform.tfstate"
    region         = "eu-south-2"
    encrypt        = true
    dynamodb_table = "tfm-slm-terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region
}
