# Infraestructura Terraform para SLM

Este directorio contiene la configuración modular de Terraform para desplegar el entorno de entrenamiento del Small Language Model (SLM).

## Estructura de Archivos

- `provider.tf`: Configuración del proveedor AWS y backend de estado en S3.
- `variables.tf`: Definición de variables (Región, Tipo de Instancia, Key Pair, etc.).
- `ecr.tf`: Repositorio de Amazon ECR para las imágenes Docker.
- `ec2.tf`: Solicitud de Instancia Spot G6, Grupos de Seguridad y Roles IAM.
- `outputs.tf`: Valores de salida (IP pública, URL del ECR).

## Requisitos Previos

1. **Backend S3:** Debes tener creado un bucket llamado `tfm-slm-terraform-state` y una tabla DynamoDB `tfm-slm-terraform-lock` en `eu-south-2` para el manejo del estado.
2. **SSH Key:** Debes tener un Key Pair llamado `tfm-slm` en la región `eu-south-2`.
3. **Secretos en GitHub:**
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

## Funcionamiento del Workflow

El flujo de trabajo de GitHub Actions (`.github/workflows/deploy.yml`) realiza lo siguiente:
1. Inicializa Terraform.
2. Crea el repositorio ECR si no existe.
3. Construye y sube la imagen Docker del proyecto.
4. Despliega una instancia **Spot G6.2xlarge** que descarga la imagen y ejecuta el entrenamiento automáticamente.
