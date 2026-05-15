import logging
from pathlib import Path

import boto3
import torch
from app.config import settings
from app.model.architecture import HybridConfig, HybridModel
from botocore.exceptions import ClientError
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def run_chat():
    """
    Interactive chat session to test the tfm-slm model from a checkpoint.
    Downloads checkpoint from S3 if not found locally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("output/checkpoint.pt")

    # Download from S3 if checkpoint not local
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client = boto3.client("s3")
        try:
            print(f"Checkpoint not found locally. Downloading from S3 ({settings.checkpoint_bucket})...")
            s3_client.download_file(
                settings.checkpoint_bucket,
                "checkpoint.pt",
                str(checkpoint_path)
            )
            print(f"✅ Checkpoint downloaded from S3")
        except ClientError as e:
            print(f"Error: No checkpoint found locally or in S3")
            print(f"Details: {e}")
            return

    # 1. Cargar Tokenizer
    print("Cargando tokenizer (GPT-2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Inicializar Arquitectura (Debe coincidir con trainer.py)
    config = HybridConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=1024,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
    )

    # 3. Cargar Pesos del Checkpoint
    print(f"Cargando pesos desde {checkpoint_path}...")
    model = HybridModel(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\n✅ Modelo cargado (Epoca: {checkpoint.get('epoch', 'unknown')})")
    print("--- Sesión de Chat Interactiva (tfm-slm) ---")
    print("Escribe 'salir' para terminar.\n")

    while True:
        user_query = input("Tú: ")
        if user_query.lower() in ["salir", "exit", "quit"]:
            break

        # Formateo básico para ayudar al modelo (Estilo Alpaca/Chat)
        prompt = f"User: {user_query}\nAssistant:"

        # Tokenización
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]

        # Generación
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=128,
                temperature=0.8,
                top_k=50
            )

        # Decodificar solo la respuesta nueva
        response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        # Limpieza básica para que no siga repitiendo "User:" en el output
        response = response.split("User:")[0].strip()

        print(f"tfm-slm: {response}")
