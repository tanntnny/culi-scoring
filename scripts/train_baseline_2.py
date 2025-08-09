import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd

LEVEL_TO_INDEX = {
    'A2_0': 0,
    'B1_1': 1,
    'B1_2': 2,
    'B2_0': 3
}

def load_texts_and_labels(folder_path):
    texts = []
    labels = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            level = '_'.join(filename.split('_')[-2:]).replace('.txt', '')
            if level not in LEVEL_TO_INDEX:
                count += 1
                #print(f"Skipping file {filename} due to unknown level {level}")
                continue
            label_index = LEVEL_TO_INDEX[level]
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label_index)
    print(f"Skipped {count} sammples due to unknown level")
    return texts, labels

folder_path = '/content/drive/MyDrive/SM_0_Unclassified_Unmerged'
texts, labels = load_texts_and_labels(folder_path)
print(f"Loaded {len(texts)} samples.")

text_series = pd.Series(texts)
text_lengths = text_series.apply(len)
mean_length_pd = text_lengths.mean()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
    }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = TextDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

from sklearn.model_selection import train_test_split

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts,
    labels,
    test_size=0.10,
    random_state=42,
    stratify=labels
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

class PrototypicalNet(nn.Module):
    def __init__(self, num_classes=4, embed_dim=256, num_prototypes=3, similarity='cosine'):
        super(PrototypicalNet, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.encoder_dim = self.encoder.config.hidden_size
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.similarity = similarity
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, 256))
        if similarity == 'cosine':
            self.s = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(0.0))
            self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.mean(dim=1)
        x = self.mlp(x)
        if self.similarity == 'cosine':
            x_norm = F.normalize(x, p=2, dim=-1)
            p_norm = F.normalize(self.prototypes, p=2, dim=-1)
            x_exp = x_norm.unsqueeze(1).unsqueeze(2)
            p_exp = p_norm.unsqueeze(0)
            sim = (x_exp * p_exp).sum(dim=-1)
            sim = sim.mean(dim=2)
            logits = (self.s * sim + self.b) / self.temp
        elif self.similarity == 'euclidean':
            dist = ((x.unsqueeze(1).unsqueeze(2) - self.prototypes.unsqueeze(0)) ** 2).sum(-1)
            sim = -dist.mean(dim=2)
            logits = sim
        return logits, x

    def init_prototypes(self, dataloader, labels):
        from sklearn.cluster import KMeans
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(next(self.parameters()).device)
                attention_mask = batch['attention_mask'].to(next(self.parameters()).device)
                outputs = self.encoder(input_ids, attention_mask)
                x = outputs.last_hidden_state.mean(1)
                x = self.mlp(x)
                embeddings.append(x.cpu())
        embeddings = torch.cat(embeddings)
        labels_tensor = torch.tensor(labels)
        for c in range(self.num_classes):
            class_embeddings = embeddings[labels_tensor == c]
            if len(class_embeddings) == 0:
                print(f"Warning: No samples found for class {c}. Initializing prototypes randomly.")
                self.prototypes.data[c] = torch.randn(self.num_prototypes, self.embed_dim).to(self.prototypes.device)
                continue
            n_clusters = min(self.num_prototypes, len(class_embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(class_embeddings.numpy())
            prototypes_c = torch.zeros(self.num_prototypes, self.embed_dim)
            prototypes_c[:n_clusters] = torch.tensor(kmeans.cluster_centers_, dtype=prototypes_c.dtype)
            self.prototypes.data[c] = prototypes_c.to(self.prototypes.device)

def initialize_prototypes(model, dataloader, num_classes=4, num_prototypes=3):
    device = next(model.parameters()).device
    class_embeddings = [[] for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Initializing Prototypes"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = model.mlp(embeddings)
            for emb, label in zip(embeddings, labels_batch):
                class_embeddings[label.item()].append(emb)
    prototype_list = []
    for class_embeds in class_embeddings:
        class_embeds = torch.stack(class_embeds)
        if len(class_embeds) < num_prototypes:
            mean_embed = class_embeds.mean(dim=0, keepdim=True)
            padded = mean_embed.repeat(num_prototypes - len(class_embeds), 1)
            proto_class = torch.cat([class_embeds, padded], dim=0)
        else:
            indices = torch.randperm(len(class_embeds))[:num_prototypes]
            proto_class = class_embeds[indices]
        prototype_list.append(proto_class)
    model.prototypes.data = torch.stack(prototype_list).to(device)
    print(f"Prototypes shape after init: {model.prototypes.shape}")

def compute_loss_weights(labels, num_classes, alpha=0.5, device='cpu'):
    label_count = torch.zeros(num_classes)
    for label in labels:
        label_count[label] += 1
    label_count = label_count.float()
    label_count_pow = label_count.pow(alpha)
    lw_weights = label_count_pow / label_count_pow.sum()
    lw_weights = lw_weights / label_count
    lw_weights = lw_weights.to(device)
    return lw_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PrototypicalNet(num_classes=4, embed_dim=256).to(device)
initialize_prototypes(model, train_dataloader, num_classes=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
lw_weights = compute_loss_weights(train_labels, num_classes=4, alpha=0.5, device=device)
criterion = nn.CrossEntropyLoss(weight=lw_weights)
train_losses = []

from sklearn.metrics import accuracy_score

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(15):
    model.train()
    total_train_loss = 0
    all_train_preds = []
    all_train_labels = []
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['label'].to(device)
        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        batch_size = logits.shape[0]
        labels_batch = labels_batch[:batch_size]
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels_batch.cpu().numpy())
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracy = accuracy_score(all_train_labels, all_train_preds) * 100
    train_losses.append(avg_train_loss)
    model.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            logits, _ = model(input_ids, attention_mask)
            batch_size = logits.shape[0]
            labels_batch = labels_batch[:batch_size]
            loss = criterion(logits, labels_batch)
            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels_batch.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds) * 100
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Training Loss: {avg_train_loss:.4f} | Training Acc: {train_accuracy:.2f}%")
    print(f"  Validation Loss: {avg_val_loss:.4f} | Validation Acc: {val_accuracy:.2f}%\n")

import matplotlib.pyplot as plt
import numpy as np

if val_accuracies:
    print(f"\nBest Val Accuracy: {max(val_accuracies):.2f}% (Epoch {np.argmax(val_accuracies)+1})")
else:
    print("\nNo validation accuracy data available to display best accuracy.")

plt.figure(figsize=(12, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax2 = plt.subplot(1, 2, 2)
ax2.plot(val_accuracies, label='Val Acc', color='green')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
plt.tight_layout()
plt.show()

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, 1)
            batch_size = input_ids.shape[0]
            labels_batch = labels_batch[:batch_size]
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    print(f"Accuracy: {correct/total*100:.2f}%")

print("\n--- Test Results ---")
if 'test_dataloader' in locals():
    evaluate(model, test_dataloader)
else:
    print("Test dataloader not found. Please perform data splitting and create test_dataloader first.")