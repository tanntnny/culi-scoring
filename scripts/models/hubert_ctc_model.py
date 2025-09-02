from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from transformers import (
    AutoModel
)

# ---------------- Hubert CTC ----------------
class HubertCTC(nn.Module):
    def __init__(
            self,
            hubert: Union[str, Path],
            vocab_size: int,
            dropout: float = 0.1,
        ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hubert)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, vocab_size)

    def forward(
            self,
            input_values: torch.Tensor,
            attention_mask: torch.Tensor = None,
        ):
        embeddings = self.encoder(input_values, attention_mask=attention_mask, output_hidden_states=False)
        embeddings = self.dropout(embeddings.last_hidden_state)
        logits = self.classifier(embeddings)
        return logits.transpose(0, 1)