from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, BertModel

from .base import ModelModule

from .attentions import CrossModalBlock
from .poolers import AttentionPooler, MeanPooler
from .encoders import PositionalEncoder

from ..registry import register

# ---------------- Cross-Modal Scorer ----------------
class CrossModalScorer(nn.Module):
    def __init__(
            self,
            num_classes: int,
            audio_encoder: Union[str, Path],
            text_encoder: Union[str, Path],
            cross_attn_hid_dim: int = 768,
            lstm_hidden_dim: int = 512,
            dropout: float = 0.0,
    ):
        super().__init__()
        
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder)
        audio_hid_dim = self.audio_encoder.config.hidden_size
        self.audio_positional_encoder = PositionalEncoder(dim_embed=audio_hid_dim, max_len=5_000)

        self.text_encoder = BertModel.from_pretrained(text_encoder)
        text_hid_dim = self.text_encoder.config.hidden_size
        self.text_positional_encoder = PositionalEncoder(dim_embed=text_hid_dim, max_len=5_000)

        self.audio_text_cross_attn = CrossModalBlock(dim_q=audio_hid_dim, dim_kv=text_hid_dim, proj_dim=cross_attn_hid_dim)
        self.text_audio_cross_attn = CrossModalBlock(dim_q=text_hid_dim, dim_kv=audio_hid_dim, proj_dim=cross_attn_hid_dim)

        self.bilstm = nn.LSTM(cross_attn_hid_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=lstm_hidden_dim * 2, num_heads=1, dropout=0.1, batch_first=True)
        self.attn_pooler = AttentionPooler(lstm_hidden_dim * 2)
        self.mean_pooler = MeanPooler()
        fc_input_dim = lstm_hidden_dim * 2 + audio_hid_dim + text_hid_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim * 4),
            nn.LayerNorm(fc_input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim * 4, fc_input_dim),
            nn.LayerNorm(fc_input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, num_classes)
        )

    def get_audio_mask(self, audio_embedding, audio_attn_mask):
         with torch.no_grad():
            input_lengths = audio_attn_mask.sum(-1)  # (B,)
            feat_lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths).to(torch.long)  # (B,)
            B, T_out, _ = audio_embedding.shape
            audio_out_mask = torch.zeros(B, T_out, device=audio_embedding.device, dtype=audio_embedding.dtype)
            for i, L in enumerate(feat_lengths):
                audio_out_mask[i, :min(L.item(), T_out)] = 1.0
            return audio_out_mask
    
    def get_text_mask(self, text_embedding, text_attn_mask):
        return text_attn_mask.to(text_embedding.dtype)

    def forward(
            self,
            audio_embedding: torch.Tensor,
            audio_attn_mask: torch.Tensor,
            text_embedding: torch.Tensor,
            text_attn_mask: torch.Tensor
    ):
        audio_embedding = self.audio_encoder(audio_embedding, audio_attn_mask).last_hidden_state
        audio_pe = self.audio_positional_encoder(audio_embedding)

        text_embedding = self.text_encoder(text_embedding, text_attn_mask).last_hidden_state
        text_pe = self.text_positional_encoder(text_embedding)

        audio_cross_attn = self.audio_text_cross_attn(audio_pe, text_pe)
        text_cross_attn = self.text_audio_cross_attn(text_pe, audio_pe)

        concat = torch.cat((audio_cross_attn, text_cross_attn), dim=1)
        lstm_out, _ = self.bilstm(concat)
        self_attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)

        audio_text_crossed, _ = self.attn_pooler(self_attn_out)
        audio_pooled = self.mean_pooler(audio_embedding, self.get_audio_mask(audio_embedding, audio_attn_mask))
        text_pooled = self.mean_pooler(text_embedding, self.get_text_mask(text_embedding, text_attn_mask))

        fusion_features = torch.cat((audio_text_crossed, audio_pooled, text_pooled), dim=1)

        z = self.fc(fusion_features)
        return z

# REGISTER

@register("model", "crossmodal")
def build_crossmodal(cfg) -> ModelModule:
    model = CrossModalScorer(
        cfg.model.num_classes,
        cfg.model.audio_encoder,
        cfg.model.text_encoder,
        cfg.model.cross_attn_hid_dim,
        cfg.model.lstm_hidden_dim,
        cfg.model.dropout,
    )
    return model