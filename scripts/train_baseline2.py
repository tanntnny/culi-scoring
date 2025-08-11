# -*- coding: utf-8 -*-

"""
Prototype-based classifier with BERT token embeddings + Bi-LSTM sequence encoder.
GPU-first: all heavy computations on GPU, AMP enabled. No mean pooling anywhere.

Requires:
  pip install torch torchvision torchaudio transformers scikit-learn tqdm pandas
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fast autotune

# ---------------------------
# Config
# ---------------------------
LEVEL_TO_INDEX = {
    'A2_0': 0,
    'B1_1': 1,
    'B1_2': 2,
    'B2_0': 3
}

DATA_FOLDER = r'C:\Users\chain\Works\Culi-scoring\SM_0_Unclassified_Unmerged-20250807T131500Z-1-001\SM_0_Unclassified_Unmerged'
PRETRAINED_MODEL = 'bert-base-uncased'
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-5
EMBED_DIM = 256
NUM_CLASSES = 4
NUM_PROTOTYPES = 3
LSTM_HIDDEN = 256
ALPHA_LOSS_WEIGHTS = 0.5
USE_AMP = True  # mixed precision for speed

# ---------------------------
# Data loading
# ---------------------------
def load_texts_and_labels(folder_path: str):
    texts, labels = [], []
    skipped = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            level = '_'.join(filename.split('_')[-2:]).replace('.txt', '')
            if level not in LEVEL_TO_INDEX:
                skipped += 1
                continue
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(LEVEL_TO_INDEX[level])
    print(f"Loaded {len(texts)} samples. Skipped {skipped} due to unknown level.")
    return texts, labels

# ---------------------------
# Dataset & Collate
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        input_ids = [b['input_ids'] for b in batch]
        attention_masks = [b['attention_mask'] for b in batch]
        labels = torch.stack([b['label'] for b in batch])

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': labels}
    return collate_fn

# ---------------------------
# Model
# ---------------------------
class PrototypicalNet(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_prototypes=NUM_PROTOTYPES,
        similarity='cosine',
        lstm_hidden=LSTM_HIDDEN,
        pretrained_name=PRETRAINED_MODEL
    ):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_name)
        self.encoder_dim = self.encoder.config.hidden_size  # 768 for bert-base-uncased

        # Bi-LSTM over token embeddings
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(
            input_size=self.encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.similarity = similarity

        # Project [B, 2*lstm_hidden] -> [B, embed_dim]
        self.mlp = nn.Sequential(
            nn.Linear(2 * lstm_hidden, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # [C, P, embed_dim]
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, embed_dim))
        if similarity == 'cosine':
            self.s = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(0.0))
            self.temp = nn.Parameter(torch.tensor(1.0))

    def _encode_with_lstm(self, input_ids, attention_mask):
        """
        Encode sequence with BERT -> BiLSTM. Uses pack_padded_sequence so padding is ignored.
        Returns tensor [B, 2*lstm_hidden].
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state  # [B, T, H]

        # lengths from attention mask (must be on CPU for pack_padded_sequence)
        lengths = attention_mask.sum(dim=1).to('cpu')
        packed = torch.nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: [2, B, lstm_hidden]
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)       # [B, 2*lstm_hidden]
        return x

    def forward(self, input_ids, attention_mask):
        x = self._encode_with_lstm(input_ids, attention_mask)  # [B, 2*lstm_hidden]
        x = self.mlp(x)                                        # [B, embed_dim]

        if self.similarity == 'cosine':
            x_norm = F.normalize(x, p=2, dim=-1)
            p_norm = F.normalize(self.prototypes, p=2, dim=-1)
            sim = (x_norm.unsqueeze(1).unsqueeze(2) * p_norm.unsqueeze(0)).sum(dim=-1).mean(dim=2)  # [B, C]
            logits = (self.s * sim + self.b) / self.temp
        else:
            dist = ((x.unsqueeze(1).unsqueeze(2) - self.prototypes.unsqueeze(0)) ** 2).sum(-1)  # [B, C, P]
            logits = -dist.mean(dim=2)  # [B, C]
        return logits, x

    def init_prototypes_kmeans(self, dataloader, all_labels):
        """
        Optional: KMeans init (CPU-only because sklearn).
        """
        from sklearn.cluster import KMeans

        device = next(self.parameters()).device
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init Prototypes (KMeans)"):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                x = self._encode_with_lstm(input_ids, attention_mask)
                x = self.mlp(x)
                embeddings.append(x.detach().cpu())  # CPU for sklearn
        embeddings = torch.cat(embeddings)  # [N, embed_dim] on CPU
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # CPU

        for c in range(self.num_classes):
            class_embeddings = embeddings[labels_tensor == c]
            if len(class_embeddings) == 0:
                print(f"Warning: No samples for class {c}. Random prototype init.")
                self.prototypes.data[c] = torch.randn(self.num_prototypes, self.embed_dim, device=device)
                continue
            n_clusters = min(self.num_prototypes, len(class_embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(class_embeddings.numpy())
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
            proto_c = torch.zeros(self.num_prototypes, self.embed_dim, device=device)
            proto_c[:n_clusters] = centers
            self.prototypes.data[c] = proto_c

# ---------------------------
# Helpers
# ---------------------------
def compute_loss_weights(labels, num_classes, alpha=0.5, device='cpu'):
    """
    Reweights CE loss roughly inversely to class frequency with smoothing power alpha.
    """
    label_count = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for y in labels:
        label_count[y] += 1
    label_count = torch.clamp(label_count, min=1.0)
    lw = (label_count.pow(alpha) / label_count.pow(alpha).sum()) / label_count
    return lw  # already on device

@torch.no_grad()
def initialize_prototypes(model, dataloader, num_classes=NUM_CLASSES, num_prototypes=NUM_PROTOTYPES, device='cuda'):
    """
    GPU path: sample embeddings per class and form prototypes.
    """
    class_embeddings = [[] for _ in range(num_classes)]
    model.eval()
    for batch in tqdm(dataloader, desc="Initializing Prototypes"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels_batch = batch['label'].to(device, non_blocking=True)

        x = model._encode_with_lstm(input_ids, attention_mask)  # [B, 2*lstm_hidden]
        x = model.mlp(x)                                        # [B, embed_dim]

        for emb, label in zip(x, labels_batch):
            class_embeddings[label.item()].append(emb)  # stays on GPU

    proto_list = []
    for class_list in class_embeddings:
        if len(class_list) == 0:
            proto_class = torch.randn(num_prototypes, model.embed_dim, device=device)
        else:
            class_tensor = torch.stack(class_list, dim=0)  # GPU
            if len(class_tensor) < num_prototypes:
                mean_embed = class_tensor.mean(dim=0, keepdim=True)
                pad = mean_embed.repeat(num_prototypes - len(class_tensor), 1)
                proto_class = torch.cat([class_tensor, pad], dim=0)
            else:
                idx = torch.randperm(len(class_tensor), device=device)[:num_prototypes]
                proto_class = class_tensor[idx]
        proto_list.append(proto_class)

    model.prototypes.data = torch.stack(proto_list, dim=0).to(device)
    print(f"Prototypes shape after init: {tuple(model.prototypes.shape)}")

def torch_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy on GPU, return python float.
    """
    correct = (preds == labels).sum(dtype=torch.float32)
    total = torch.tensor(labels.numel(), dtype=torch.float32, device=labels.device)
    acc = (correct / torch.clamp(total, min=1.0)) * 100.0
    return float(acc.item())

# ---------------------------
# Main
# ---------------------------
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    texts, labels = load_texts_and_labels(DATA_FOLDER)
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

    # Splits
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.10, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset   = TextDataset(val_texts,   val_labels,   tokenizer)
    test_dataset  = TextDataset(test_texts,  test_labels,  tokenizer)

    # Dataloaders (pin_memory helps GPU transfer)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=pin)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=pin)

    # Model
    model = PrototypicalNet(
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_prototypes=NUM_PROTOTYPES,
        similarity='cosine',
        lstm_hidden=LSTM_HIDDEN,
        pretrained_name=PRETRAINED_MODEL
    ).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lw_weights = compute_loss_weights(train_labels, num_classes=NUM_CLASSES, alpha=ALPHA_LOSS_WEIGHTS, device=device)
    criterion = nn.CrossEntropyLoss(weight=lw_weights)

    # Prototype init (GPU path)
    initialize_prototypes(model, train_loader, num_classes=NUM_CLASSES, num_prototypes=NUM_PROTOTYPES, device=device)
    # If you want KMeans init instead (CPU), call:
    # model.init_prototypes_kmeans(train_loader, train_labels)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct_sum = 0.0
        total_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            input_ids     = batch['input_ids'].to(device, non_blocking=True)
            attention_mask= batch['attention_mask'].to(device, non_blocking=True)
            labels_batch  = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, _ = model(input_ids, attention_mask)
                labels_batch = labels_batch[:logits.shape[0]]
                loss = criterion(logits, labels_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().item())
            preds = torch.argmax(logits, dim=1)
            correct_sum += (preds == labels_batch).sum(dtype=torch.float32)
            total_count += labels_batch.numel()

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = float((correct_sum / max(total_count, 1)).item() * 100.0)

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        v_correct = 0.0
        v_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                input_ids     = batch['input_ids'].to(device, non_blocking=True)
                attention_mask= batch['attention_mask'].to(device, non_blocking=True)
                labels_batch  = batch['label'].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logits, _ = model(input_ids, attention_mask)
                    labels_batch = labels_batch[:logits.shape[0]]
                    loss = criterion(logits, labels_batch)

                val_loss_sum += float(loss.detach().item())
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == labels_batch).sum(dtype=torch.float32)
                v_total += labels_batch.numel()

        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = float((v_correct / max(v_total, 1)).item() * 100.0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%\n")

    # ---- Test ----
    print("\n--- Test Results ---")
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    t_correct = 0.0
    t_total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids     = batch['input_ids'].to(device, non_blocking=True)
            attention_mask= batch['attention_mask'].to(device, non_blocking=True)
            labels_batch  = batch['label'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, _ = model(input_ids, attention_mask)
                labels_batch = labels_batch[:logits.shape[0]]

            preds = torch.argmax(logits, dim=1)
            t_correct += (preds == labels_batch).sum(dtype=torch.float32)
            t_total += labels_batch.numel()

    test_acc = float((t_correct / max(t_total, 1)).item() * 100.0)
    print(f"Accuracy: {test_acc:.2f}%")

    # Sanity prints
    print("\nSanity check:")
    print("MLP in_features (should be 2*lstm_hidden):", model.mlp[0].in_features)
    print("Prototype tensor shape:", tuple(model.prototypes.shape))

if __name__ == "__main__":
    main()
