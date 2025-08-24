from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import math
from pathlib import Path

import torch
import torch.nn as nn

from scripts.models.transformer_essentials import (
    PositionalEncoder,
    AttentionPooler,
)
from transformers import Wav2Vec2Model, BertModel

# ---------------- Mean Pooler ----------------
class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

class PrototypicalClassifier(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 k: int = 3,
                 metric: str = 'sed'
                 ):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.metric = metric.lower()
        self.prototypes = nn.Parameter(torch.randn(num_classes * k, embed_dim) / math.sqrt(embed_dim))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.metric == 'cos':
            # Cosine similarity
            x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
            proto_norm = self.prototypes / (self.prototypes.norm(dim=-1, keepdim=True) + 1e-9)
            sims = torch.matmul(x_norm, proto_norm.t())  # (B, num_classes*k)
            sims = sims.view(x.size(0), self.num_classes, self.k)
            sims = sims.mean(dim=2)
            logits = sims / torch.exp(self.log_tau)
            return logits
        else:
            # Squared Euclidean Distance (SED)
            dists = torch.cdist(x, self.prototypes, p=2).pow(2)
            dists = dists.view(x.size(0), self.num_classes, self.k)
            dists = dists.mean(dim=2)
            logits = -dists / torch.exp(self.log_tau)
            return logits

# Speech Model
class SpeechModel(nn.Module):
    def __init__(self, num_classes: int, k: int = 3, embed_dim: int = 256, metric: str = 'sed'):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("models/wav2vec2-model")
        hidden_size = self.encoder.config.hidden_size
        self.pooler = MeanPooler()
        self.project = nn.Sequential(
            nn.Linear(hidden_size, 4024),
            nn.GELU(),
            nn.Linear(4024, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.metric_head = PrototypicalClassifier(embed_dim=embed_dim, num_classes=num_classes, k=k, metric=metric)

    def _get_feature_level_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = attention_mask.unsqueeze(1).float()  # (B,1,T)
            for conv in self.encoder.feature_extractor.conv_layers:
                x = conv(x)
            mask = (x.abs() > 0).any(dim=1).long()  # (B, T_feat)
            return mask

    def _module_device(self):
        return next(self.parameters()).device
    
    def _cast_audio_inputs(self, audio_embeddings):
        device = self._module_device()
        enc_dtype = next(self.encoder.parameters()).dtype
        return {
            "input_values": audio_embeddings["input_values"].to(device, dtype=enc_dtype, non_blocking=True),
            "attention_mask": audio_embeddings["attention_mask"].to(device, dtype=torch.long, non_blocking=True),
        }

    def forward(self, audio_embeddings, **kwargs) -> torch.Tensor:
        audio_embeddings = self._cast_audio_inputs(audio_embeddings)
        out = self.encoder(input_values=audio_embeddings["input_values"], attention_mask=audio_embeddings["attention_mask"])
        hidden = out.last_hidden_state  # (B, T_feat, H)
        feat_mask = self._get_feature_level_mask(audio_embeddings["attention_mask"])  # (B, T_feat)
        pooled = self.pooler(hidden, feat_mask)
        z = self.project(pooled)  # (B, 256)
        logits = self.metric_head(z)
        return logits

# Multimodal Model
class IndividualModalModel(nn.Module):
    def __init__(self,
                wav2vec2_encoder: Path,
                text_encoder: Path,
                num_classes: int,
                k: int = 3,
                lstm_hidden_dim: int = 256,
                fusion_proj_dim: int = 256,
                pt_metric: str = "sed"
                ):
        super().__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec2_encoder)
        # self.audio_encoder.gradient_checkpointing_enable()
        self.audio_encoder.feature_extractor.to(dtype=torch.float32)
        self.audio_lstm = nn.LSTM(self.audio_encoder.config.hidden_size, lstm_hidden_dim, batch_first=True, bidirectional=True)

        self.text_encoder = BertModel.from_pretrained(text_encoder, gradient_checkpointing=False)
        # self.text_encoder.gradient_checkpointing_enable()
        self.text_lstm = nn.LSTM(self.text_encoder.config.hidden_size, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.fusion_projection = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 4, lstm_hidden_dim * 2), # (2 * Audio + 2 * Text)
            nn.GELU(),
            nn.Linear(lstm_hidden_dim * 2, fusion_proj_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_proj_dim)
        )
        self.metric_head = PrototypicalClassifier(embed_dim=fusion_proj_dim, num_classes=num_classes, k=k, metric=pt_metric)

    def _module_device(self):
        return next(self.parameters()).device
    
    def _cast_audio_inputs(self, audio_embeddings):
        device = self._module_device()
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        return {
            "input_values": audio_embeddings["input_values"].to(device, dtype=enc_dtype, non_blocking=True),
            "attention_mask": audio_embeddings["attention_mask"].to(device, dtype=torch.long, non_blocking=True),
        }

    def _cast_text_inputs(self, text_embeddings):
        device = self._module_device()
        return {
            "input_ids": text_embeddings["input_ids"].to(device, dtype=torch.long, non_blocking=True),
            "attention_mask": text_embeddings["attention_mask"].to(device, dtype=torch.long, non_blocking=True),
        }

    def _get_feature_level_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            input_lengths = attention_mask.sum(dim=1)
            conv_lengths = (input_lengths - 400) // 320 + 1
            conv_lengths = torch.clamp(conv_lengths, min=1)
            max_len = conv_lengths.max().item()
            mask = torch.zeros(
                len(conv_lengths),
                max_len,
                dtype=torch.long,
                device=attention_mask.device
            )
            for i, l in enumerate(conv_lengths):
                mask[i, :l] = 1
            return mask

    def forward(self, audio_embeddings, text_embeddings, **kwargs):
        # 1) Make devices/dtypes consistent
        audio_embeddings = self._cast_audio_inputs(audio_embeddings)
        text_embeddings  = self._cast_text_inputs(text_embeddings)

        # 2) AUDIO (encoder -> LSTM -> masked mean pool)
        audio_outputs = self.audio_encoder(**audio_embeddings)
        audio_hidden_states = audio_outputs.last_hidden_state  # (B, T_feat, H)
        B, T_feat, _ = audio_hidden_states.shape
        device = audio_hidden_states.device
        adtype = audio_hidden_states.dtype

        # feature-level mask (B, T_feat) in {0,1}
        feature_level_mask = self._get_feature_level_mask(audio_embeddings["attention_mask"])  # long/bool
        audio_feature_lengths = feature_level_mask.sum(dim=1)  # (B,)

        # buffers that match dtype/device of the hidden states
        audio_pooled = audio_hidden_states.new_zeros(B, self.audio_lstm.hidden_size * 2)

        valid_audio_indices = audio_feature_lengths > 0
        if valid_audio_indices.any():
            valid_audio_hidden  = audio_hidden_states[valid_audio_indices]               # (Bv, T_feat, H)
            valid_audio_lengths = audio_feature_lengths[valid_audio_indices]             # (Bv,)
            valid_feature_mask  = feature_level_mask[valid_audio_indices]                # (Bv, T_feat)

            audio_packed = nn.utils.rnn.pack_padded_sequence(
                valid_audio_hidden, valid_audio_lengths.to("cpu"), batch_first=True, enforce_sorted=False
            )
            audio_lstm_out_packed, _ = self.audio_lstm(audio_packed)
            audio_lstm_out_valid, _ = nn.utils.rnn.pad_packed_sequence(audio_lstm_out_packed, batch_first=True)
            # pool in fp32 for numerical stability, then cast back
            vf_mask = valid_feature_mask.unsqueeze(-1).float()
            summed_audio  = (audio_lstm_out_valid.float() * vf_mask).sum(dim=1)
            counts_audio  = vf_mask.sum(dim=1).clamp(min=1e-6)
            pooled_audio_valid = (summed_audio / counts_audio).to(adtype)
            audio_pooled[valid_audio_indices] = pooled_audio_valid
        else:
            # no valid sequences; leave zeros
            pass

        # 3) TEXT (encoder -> LSTM -> masked mean pool)
        text_outputs = self.text_encoder(**text_embeddings)
        text_hidden_states = text_outputs.last_hidden_state  # (B, T_txt, Ht)
        B, T_txt, _ = text_hidden_states.shape
        tdtype = text_hidden_states.dtype

        text_lengths = text_embeddings["attention_mask"].sum(dim=1)  # (B,)
        text_pooled = text_hidden_states.new_zeros(B, self.text_lstm.hidden_size * 2)

        valid_text_indices = text_lengths > 0
        if valid_text_indices.any():
            valid_text_hidden = text_hidden_states[valid_text_indices]                # (Bv, T_txt, Ht)
            valid_text_lengths = text_lengths[valid_text_indices]                     # (Bv,)
            valid_text_mask = text_embeddings["attention_mask"][valid_text_indices]   # (Bv, T_txt)

            text_packed = nn.utils.rnn.pack_padded_sequence(
                valid_text_hidden, valid_text_lengths.to("cpu"), batch_first=True, enforce_sorted=False
            )
            text_lstm_out_packed, _ = self.text_lstm(text_packed)
            text_lstm_out_valid, _ = nn.utils.rnn.pad_packed_sequence(text_lstm_out_packed, batch_first=True)

            vt_mask = valid_text_mask.unsqueeze(-1).float()
            summed_text = (text_lstm_out_valid.float() * vt_mask).sum(dim=1)
            counts_text = vt_mask.sum(dim=1).clamp(min=1e-6)
            pooled_text_valid = (summed_text / counts_text).to(tdtype)
            text_pooled[valid_text_indices] = pooled_text_valid
        else:
            pass

        # 4) FUSION
        combined_features = torch.cat((audio_pooled, text_pooled), dim=1)  # (B, 4 * lstm_hidden_dim)
        fused_features = self.fusion_projection(combined_features)
        
        # 5) CLASSIFIER (run head in fp32 for stable distances)
        logits = self.metric_head(fused_features.float())
        return logits

# ---------------- Cross-Modal Block ----------------
class CrossModalBlock(nn.Module):
    def __init__(
            self,
            dim_q: int,
            dim_kv: int,
            proj_dim: int,
            n_heads: int = 8,
            dropout: float = 0.1,
    ):
        super().__init__()
        assert dim_q % n_heads == 0, "dim_q must be divisible by n_heads"

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            kdim=dim_kv,
            vdim=dim_kv,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(dim_q, dim_q * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_q * 4, dim_q)
        )
        self.norm_q_1 = nn.LayerNorm(dim_q)
        self.norm_q_2 = nn.LayerNorm(dim_q)
        self.norm_q_3 = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        self.proj = nn.Linear(dim_q, proj_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
    ):
        # q shape : (Batch, Sequence_q, Feature_q)
        # kv shape : (Batch, Sequence_kv, Feature_kv)
        q_norm = self.norm_q_1(q)
        kv_norm = self.norm_kv(kv)
        x, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        q = q + self.dropout(x)
        
        q_norm = self.norm_q_2(q)
        q = q + self.dropout(self.ff(q_norm))

        q_norm = self.norm_q_3(q)
        x, _ = self.self_attn(q_norm, q_norm, q_norm)
        q = q + self.dropout(x)

        z = self.proj(q)

        return z

# ---------------- Linguistic Feature Block ----------------
# TODO

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

        self.bilstm = nn.LSTM(cross_attn_hid_dim * 2, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=lstm_hidden_dim * 2, num_heads=1, dropout=0.1, batch_first=True)
        self.attn_pooler = AttentionPooler(lstm_hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_classes)
        )

    def forward(
            self,
            audio_embedding: torch.Tensor,
            audio_attn_mask: torch.Tensor,
            text_embedding: torch.Tensor,
            text_attn_mask: torch.Tensor
    ):
        audio_embedding = self.audio_encoder(audio_embedding, audio_attn_mask).last_hidden_state
        audio_embedding = self.audio_positional_encoder(audio_embedding)

        text_embedding = self.text_encoder(text_embedding, text_attn_mask).last_hidden_state
        text_embedding = self.text_positional_encoder(text_embedding)

        audio_embedding = self.audio_text_cross_attn(audio_embedding, text_embedding)
        text_embedding = self.text_audio_cross_attn(text_embedding, audio_embedding)

        concat = torch.cat((audio_embedding, text_embedding), dim=-1)
        lstm_out, _ = self.bilstm(concat)
        self_attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        pooled = self.attn_pooler(self_attn_out)

        z = self.fc(pooled)
        return z