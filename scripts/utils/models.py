import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, BertModel

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


    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T_feat, H)
        feat_mask = self._get_feature_level_mask(attention_mask)  # (B, T_feat)
        pooled = self.pooler(hidden, feat_mask)
        z = self.project(pooled)  # (B, 256)
        logits = self.metric_head(z)
        return logits

# Multimodal Model
class MultimodalModel(nn.Module):
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
        self.mlp_head = nn.Sequential(
            nn.Linear(fusion_proj_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
            nn.LayerNorm(num_classes)
        )

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
        dev = self._module_device()
        return {
            "input_ids": text_embeddings["input_ids"].to(dev, dtype=torch.long, non_blocking=True),
            "attention_mask": text_embeddings["attention_mask"].to(dev, dtype=torch.long, non_blocking=True),
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

    def forward(self, audio_embeddings, text_embeddings, ids, labels):
        # 1) Make devices/dtypes consistent
        audio_embeddings = self._cast_audio_inputs(audio_embeddings)
        text_embeddings  = self._cast_text_inputs(text_embeddings)

        # 2) AUDIO (encoder -> LSTM -> masked mean pool)
        audio_outputs = self.audio_encoder(
            input_values=audio_embeddings["input_values"],
            attention_mask=audio_embeddings["attention_mask"]
        )
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
        text_outputs = self.text_encoder(
            input_ids=text_embeddings["input_ids"],
            attention_mask=text_embeddings["attention_mask"]
        )
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
        
        logits = self.mlp_head(fused_features.float())

        # 5) CLASSIFIER (run head in fp32 for stable distances)
        # logits = self.metric_head(fused_features.float())
        return logits