# Nexus-SLM: Modelo de Lenguaje Pequeño con Arquitectura Híbrida Transformer-GRU

Nexus-SLM es un modelo de lenguaje de escala reducida (Small Language Model) desarrollado en el marco de un Trabajo de Fin de Máster (TFM). El proyecto implementa una arquitectura híbrida personalizada que integra mecanismos de atención global (Transformers) con la eficiencia del refinamiento secuencial (GRU), optimizando el rendimiento para hardware NVIDIA de última generación y flujos de trabajo MLOps profesionales.

## Características Principales

*   Arquitectura Híbrida Integrada: 12 capas compuestas por bloques híbridos (HybridBlock) que fusionan Multi-Head Attention y GRU.
*   Eficiencia en Parámetros: Configuración de aproximadamente 124M de parámetros con implementación de Weight Tying entre embeddings y la cabeza de salida.
*   Entrenamiento de Alto Rendimiento: Soporte nativo para precisión bfloat16 y aceleración de matmuls mediante TF32, optimizado específicamente para GPUs NVIDIA L4 (Arquitectura Ada Lovelace).
*   Pipeline de Datos Profesional: Servicios desacoplados siguiendo principios SOLID, utilizando el formato Apache Arrow para una gestión eficiente de la memoria mediante memory-mapping.
*   MLOps Determinista: Entornos reproducibles mediante el gestor uv, construcción de contenedores optimizada con caché global y CI/CD dirigido por Pull Requests.

## Arquitectura del Modelo: HybridBlock

A diferencia de las arquitecturas híbridas convencionales que se limitan a concatenar capas de distinto tipo, Nexus-SLM propone una integración profunda en cada nivel del modelo a través del componente HybridBlock. Cada uno de los 12 bloques del sistema ejecuta la siguiente secuencia de procesamiento:

1.  Normalización Previa (Pre-LayerNorm): Aplicada antes de cada sub-componente para garantizar la estabilidad numérica durante el entrenamiento de la arquitectura desde cero.
2.  Auto-Atención Multi-Cabeza (MHA): Configurada con 12 cabezas de atención, se encarga de capturar dependencias globales y relaciones semánticas de largo alcance en secuencias de hasta 1024 tokens.
3.  Unidad Recurrente Puerta (GRU): Situada inmediatamente después de la atención, actúa como un filtro de refinamiento secuencial. Esta integración permite al modelo mantener un estado interno persistente para patrones locales y estructurales, optimizando la representación del lenguaje sin el coste cuadrático de capas de atención adicionales.
4.  Red de Realimentación (FFN): Una estructura MLP con factor de expansión 4 y activación GELU para el procesamiento de características de alto nivel.

El diseño se completa con una estrategia de Weight Tying, que reduce el uso de memoria de video (VRAM) en un 18% al compartir pesos entre la capa de entrada y salida, permitiendo batch sizes más elevados durante el entrenamiento.

## Estrategia de Mezcla de Datos (Data Mixing)

El modelo utiliza una combinación estratégica de cinco pilares de datos abiertos:
*   Conversacional (50%): OpenAssistant y UltraChat, para asegurar naturalidad en el diálogo y consistencia en contextos largos.
*   Instrucciones (50%): Alpaca y ShareGPT, enfocados en dotar al modelo de capacidades resolutivas y seguimiento de instrucciones.
*   Especializado: Subconjunto de The Stack (YAML) para el conocimiento de sintaxis de infraestructura y flujos DevOps.

## Instalación y Uso

### Requisitos Previos
*   uv (Gestor de paquetes de Python)
*   Docker (Opcional, para ejecución en contenedores)
*   GPU NVIDIA con 24GB de VRAM (Recomendado: instancia g6.xlarge de AWS)

### Configuración Local
```bash
# Sincronizar el entorno e instalar dependencias
uv sync

# Ejecutar el pipeline completo (Descarga -> Procesamiento -> Entrenamiento)
uv run tfm-slm
```

### Docker
```bash
# Construcción de la imagen optimizada
docker build -t nexus-slm:latest .

# Ejecución del contenedor con soporte de GPU
docker run --rm --gpus all nexus-slm:latest
```

## Estructura del Proyecto

*   app/: Código fuente principal en Python.
    *   model/: Implementación técnica de la arquitectura híbrida.
    *   training/: Servicio de entrenamiento optimizado para NVIDIA L4.
    *   dataset/: Servicios de adquisición, armonización y mezcla de datos.
*   report/: Fuentes en LaTeX de la memoria del TFM (Capítulos 1-7).
*   .github/workflows/: Flujos de CI/CD actualizados a Node.js 24 con estrategia de Caché Global.

## Memoria y Documentación
La documentación técnica detallada se encuentra en el directorio report/. El reporte incluye el análisis de reproducibilidad (uv.lock), eficiencia en costes cloud y el estudio comparativo frente a la familia de modelos SmolLM de Hugging Face.
