import torch
from app.model.architecture import HybridModel, HybridConfig
from transformers import AutoTokenizer
from pathlib import Path

def run_chat():
    """
    Interactive chat session to test the tfm-slm model from a checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("output/checkpoint.pt")

    if not checkpoint_path.exists():
        print(f"Error: No se encuentra el archivo {checkpoint_path}")
        print("Asegúrate de que el entrenamiento ha generado al menos un checkpoint.")
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

if __name__ == "__main__":
    run_chat()
