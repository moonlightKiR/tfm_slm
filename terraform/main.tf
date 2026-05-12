provider "aws" {
  region = var.aws_region
}

# --- CONFIGURACIÓN DE ESTADO ---
terraform {
  backend "s3" {
    bucket = "tfm-slm-terraform-state-guille"
    key    = "training/terraform.tfstate"
    region = "eu-south-2"
  }
}

variable "aws_region" {
  default = "eu-south-2"
}

variable "image_tag" {
  description = "Tag de la imagen en ECR"
}

variable "hf_token" {
  description = "Hugging Face Token para datasets/modelos"
  default     = ""
}

# --- INFRAESTRUCTURA IAM ---

resource "aws_iam_role" "training_role" {
  name = "TFM-SLM-Training-Role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_policy" {
  role       = aws_iam_role.training_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "ecr_policy" {
  role       = aws_iam_role.training_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "training_profile" {
  name = "TFM-SLM-Instance-Profile"
  role = aws_iam_role.training_role.name
}

# --- INSTANCIA GPU SPOT ---

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-ecs-gpu-hvm-2023.*-x86_64"]
  }
}

resource "aws_instance" "training_spot" {
  ami           = data.aws_ami.dlami.id
  instance_type = "g6.xlarge"
  
  # AHORRO: La instancia se destruye automáticamente al apagarse
  instance_initiated_shutdown_behavior = "terminate"

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = "0.60"
    }
  }

  iam_instance_profile = aws_iam_instance_profile.training_profile.name
  
  user_data = <<-EOF
              #!/bin/bash
              while ! docker info > /dev/null 2>&1; do sleep 1; done
              
              # Login en ECR
              aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${data.aws_ecr_repository.repo.repository_url}
              
              # Ejecutar entrenamiento (inyectando HF_TOKEN si existe)
              docker run --gpus all \
                -e HF_TOKEN=${var.hf_token} \
                ${data.aws_ecr_repository.repo.repository_url}:${var.image_tag}
              
              # AHORRO: Apagar la máquina al terminar para que Terraform/AWS la eliminen
              shutdown -h now
              EOF

  tags = {
    Name = "TFM-SLM-Spot-Training"
  }
}

data "aws_ecr_repository" "repo" {
  name = "tfm-slm"
}
