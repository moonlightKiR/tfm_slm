# Guía Rápida de Comandos (Cheat Sheet)

Esta guía contiene los comandos esenciales para gestionar la infraestructura, la conexión y el entrenamiento del SLM.

## 1. Gestión de Infraestructura (Terraform)
*Ejecutar desde la carpeta `app/terraform/` en tu ordenador local.*

- **Inicializar**: `terraform init`
- **Ver cambios pendientes**: `terraform plan`
- **Desplegar cambios**: `terraform apply -auto-approve`
- **Obtener la IP de la máquina**: `terraform output ec2_public_ip`
- **Destruir todo (Ahorro de costes)**: `terraform destroy -auto-approve`
- **Importar recurso existente**: `terraform import <direccion_recurso> <id_en_aws>`
- **Desbloquear estado (Lock error)**: `terraform force-unlock <LOCK_ID>`

## 2. Conexión Remota (SSH)
*Ejecutar desde la carpeta donde tengas el archivo `.pem`.*

- **Configurar permisos (solo la primera vez)**: `chmod 400 tfm-slm.pem`
- **Conectar a la instancia**: `ssh -i "tfm-slm.pem" ec2-user@<IP_PUBLICA>`
- **Limpiar IP antigua (si da error de host)**: `ssh-keygen -R <IP_PUBLICA>`

## 3. Control del Entrenamiento (Docker)
*Ejecutar una vez dentro de la instancia EC2 por SSH.*

- **Ver contenedores activos**: `docker ps`
- **Ver logs del entrenamiento (en vivo)**: `docker logs -f <ID_CONTENEDOR>`
- **Ver todos los contenedores (incluidos fallidos)**: `docker ps -a`
- **Ver logs de un contenedor que ya falló**: `docker logs <ID_CONTENEDOR>`
- **Borrar todos los contenedores parados**: `docker container prune -f`

## 4. Monitorización de Hardware (NVIDIA)
*Ejecutar una vez dentro de la instancia EC2 por SSH.*

- **Ver estado de la GPU y VRAM**: `nvidia-smi`
- **Monitorizar GPU cada segundo**: `nvidia-smi -l 1`

## 5. Depuración de Arranque (Cloud-Init)
*Si la máquina acaba de encenderse y `docker ps` no muestra nada.*

- **Ver progreso de instalación inicial y descarga de imagen**: `sudo tail -f /var/log/cloud-init-output.log`

## 6. Gestión de Sesiones (Pausar y Reanudar)
*Permite ahorrar costes deteniendo la máquina sin perder el progreso del entrenamiento.*

- **Pausar Entrenamiento (Solo EC2)**:
  `terraform destroy -target=aws_spot_instance_request.training -auto-approve`
  *(Detiene la máquina. S3 + ECR mantienen checkpoints e imagen Docker).*

- **Verificar que se paró**:
  `terraform output ec2_public_ip`
  *(Si devuelve vacío, la instancia se destruyó correctamente).*

- **Reanudar Entrenamiento**:
  `terraform apply -auto-approve`
  *(Nueva máquina descarga checkpoint de S3 y continúa desde la época guardada).*
