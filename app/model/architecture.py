import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class HybridConfig(PretrainedConfig):
    model_type = "hybrid_transformer_gru"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout


class HybridModel(PreTrainedModel):
    config_class = HybridConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Transformer Layers (Encoder-style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # GRU Refinement Layer
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.ln_final = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(self, input_ids, labels=None):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0)
        x = self.embeddings(input_ids) + self.pos_embeddings(positions)

        # Transformer blocks (Global context)
        # Fix: generate_square_subsequent_mask is the correct method in PyTorch
        mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(device)
        x = self.transformer_encoder(x, mask=mask, is_causal=True)

        # GRU (Sequential refinement)
        x, _ = self.gru(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
