# Configuración del Backend de Terraform en AWS

Ejecuta estos comandos en tu terminal local para crear los recursos necesarios para guardar el estado de Terraform de forma segura.

### 1. Crear el Bucket de S3
Este bucket almacenará el archivo `terraform.tfstate`.

```bash
aws s3api create-bucket \
    --bucket tfm-slm-terraform-state \
    --region eu-south-2 \
    --create-bucket-configuration LocationConstraint=eu-south-2
```

### 2. Habilitar el Versionado
Esto permite recuperar versiones anteriores del estado si algo falla.

```bash
aws s3api put-bucket-versioning \
    --bucket tfm-slm-terraform-state \
    --versioning-configuration Status=Enabled
```

---

## Nota sobre el Bloqueo (Locking)
Hemos configurado el proyecto para usar **S3 Native Locking** (`use_lockfile = true`). 
Esto significa que ya **no necesitas la tabla de DynamoDB** que creamos anteriormente. Terraform usará el propio bucket de S3 para gestionar los bloqueos de forma segura y moderna.

Si ya creaste la tabla de DynamoDB, puedes dejarla ahí o borrarla con:
```bash
aws dynamodb delete-table --table-name tfm-slm-terraform-lock --region eu-south-2
```
