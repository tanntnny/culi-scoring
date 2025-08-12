import torch
import torchaudio
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, BertTokenizer, Wav2Vec2Model, BertModel
import torch.nn.utils.rnn
import math
import torch.nn as nn
from tqdm import tqdm, notebook
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def audio_to_tensor(path, frame_rate=16_000, min_length=16000):
    try:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != frame_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=frame_rate)
            waveform = resampler(waveform)
        waveform = waveform.squeeze().numpy()
        if waveform.shape[0] < min_length:
            print(f"Warning: Audio file {path} is too short (length={waveform.shape[0]}). Padding with zeros.")
            padding = np.zeros(min_length - waveform.shape[0])
            waveform = np.concatenate((waveform, padding))
        return waveform, frame_rate
    except Exception as e:
        print(f"Error loading or processing audio file {path}: {e}")
        return np.zeros(min_length), frame_rate

class MultimodalDataset(Dataset):
    def __init__(self, audio_paths, text_paths, labels, audio_processor, text_tokenizer):
        self.audio_paths = audio_paths
        self.text_paths = text_paths
        self.labels = labels
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        text_path = self.text_paths[idx]
        label = self.labels[idx]
        waveform, _ = audio_to_tensor(audio_path)
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text_encoding = self.text_tokenizer(
            text,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'audio': waveform,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def multimodal_collate_fn(batch):
    audio_list = [item['audio'] for item in batch]
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['label'] for item in batch]
    processed_audio = wav2vec_processor(
        audio_list,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    audio_input_values = processed_audio["input_values"]
    audio_attention_mask = processed_audio["attention_mask"]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_batch = torch.stack(labels_list)
    return {
        'audio_input_values': audio_input_values,
        'audio_attention_mask': audio_attention_mask,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'label': labels_batch
    }

class PrototypicalClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, k: int = 3):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.prototypes = nn.Parameter(
            torch.randn(num_classes * k, embed_dim) / math.sqrt(embed_dim)
        )
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.prototypes, p=2).pow(2)
        dists = dists.view(x.size(0), self.num_classes, self.k)
        dists = dists.mean(dim=2)
        logits = -dists / torch.exp(self.log_tau)
        return logits

class MultimodalModel(nn.Module):
    def __init__(self, num_classes: int, k: int = 3, audio_embed_dim: int = 256, text_embed_dim: int = 256, combined_embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.audio_encoder.gradient_checkpointing_enable()
        self.audio_lstm = nn.LSTM(self.audio_encoder.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.audio_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, audio_embed_dim),
            nn.GELU(),
            nn.LayerNorm(audio_embed_dim)
        )
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder.gradient_checkpointing_enable()
        self.text_lstm = nn.LSTM(self.text_encoder.config.hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.text_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, text_embed_dim),
            nn.GELU(),
            nn.LayerNorm(text_embed_dim)
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(audio_embed_dim + text_embed_dim, combined_embed_dim),
            nn.GELU(),
            nn.LayerNorm(combined_embed_dim)
        )
        self.metric_head = PrototypicalClassifier(embed_dim=combined_embed_dim, num_classes=num_classes, k=k)

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

    def forward(self, audio_input_values: torch.Tensor, audio_attention_mask: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size = audio_input_values.size(0)
        device = audio_input_values.device
        audio_outputs = self.audio_encoder(input_values=audio_input_values, attention_mask=audio_attention_mask)
        audio_hidden_states = audio_outputs.last_hidden_state
        feature_level_mask = self._get_feature_level_mask(audio_attention_mask)
        assert audio_hidden_states.shape[1] == feature_level_mask.shape[1], \
            f"Mismatch! Features:{audio_hidden_states.shape[1]} vs Mask:{feature_level_mask.shape[1]}"
        audio_feature_lengths = torch.tensor(
            [audio_hidden_states.shape[1]] * len(audio_hidden_states),
            device=device
        )
        audio_lstm_out = torch.zeros(batch_size, audio_hidden_states.size(1), self.audio_lstm.hidden_size * 2, device=device)
        audio_pooled = torch.zeros(batch_size, self.audio_lstm.hidden_size * 2, device=device)
        valid_audio_indices = audio_feature_lengths > 0
        if valid_audio_indices.any():
            valid_audio_hidden = audio_hidden_states[valid_audio_indices]
            valid_audio_lengths = audio_feature_lengths[valid_audio_indices]
            valid_feature_mask = feature_level_mask[valid_audio_indices]
            audio_packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
                valid_audio_hidden,
                valid_audio_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            audio_lstm_out_packed, _ = self.audio_lstm(audio_packed_sequences)
            audio_lstm_out_valid, _ = torch.nn.utils.rnn.pad_packed_sequence(audio_lstm_out_packed, batch_first=True)
            audio_lstm_out[valid_audio_indices] = audio_lstm_out_valid
            summed_audio = (audio_lstm_out_valid * valid_feature_mask.unsqueeze(-1).float()).sum(dim=1)
            counts_audio = valid_feature_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9).float()
            valid_audio_pooled = summed_audio / counts_audio
            audio_pooled[valid_audio_indices] = valid_audio_pooled
        else:
            print("\n[Warning] No valid audio sequences in this batch!")
        audio_features = self.audio_projection(audio_pooled)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden_states = text_outputs.last_hidden_state
        text_lengths = attention_mask.sum(dim=1)
        text_lstm_out = torch.zeros(batch_size, text_hidden_states.size(1), self.text_lstm.hidden_size * 2, device=device)
        text_pooled = torch.zeros(batch_size, self.text_lstm.hidden_size * 2, device=device)
        valid_text_indices = text_lengths > 0
        if valid_text_indices.any():
            valid_text_hidden = text_hidden_states[valid_text_indices]
            valid_text_lengths = text_lengths[valid_text_indices]
            valid_text_mask = attention_mask[valid_text_indices]
            text_packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(
                valid_text_hidden,
                valid_text_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            text_lstm_out_packed, _ = self.text_lstm(text_packed_sequences)
            text_lstm_out_valid, _ = torch.nn.utils.rnn.pad_packed_sequence(text_lstm_out_packed, batch_first=True)
            text_lstm_out[valid_text_indices] = text_lstm_out_valid
            summed_text = (text_lstm_out_valid * valid_text_mask.unsqueeze(-1).float()).sum(dim=1)
            counts_text = valid_text_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9).float()
            valid_text_pooled = summed_text / counts_text
            text_pooled[valid_text_indices] = valid_text_pooled
        text_features = self.text_projection(text_pooled)
        combined_features = torch.cat((audio_features, text_features), dim=1)
        fused_features = self.fusion_projection(combined_features)
        logits = self.metric_head(fused_features)
        return logits, fused_features

    def init_prototypes(self, dataloader, num_classes, num_prototypes, max_batches: int = None):
        device = next(self.parameters()).device
        class_features = [[] for _ in range(num_classes)]
        self.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Initializing Prototypes")):
                if max_batches is not None and i >= max_batches:
                    print(f"\nReached max_batches limit ({max_batches}) for prototype initialization.")
                    break
                audio_input_values = batch['audio_input_values'].to(device)
                audio_attention_mask = batch['audio_attention_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['label'].to(device)
                valid_audio = audio_attention_mask.sum(dim=1) > 0
                valid_text = attention_mask.sum(dim=1) > 0
                valid_samples_mask = valid_audio & valid_text
                audio_input_values = audio_input_values[valid_samples_mask]
                audio_attention_mask = audio_attention_mask[valid_samples_mask]
                input_ids = input_ids[valid_samples_mask]
                attention_mask = attention_mask[valid_samples_mask]
                labels_batch = labels_batch[valid_samples_mask]
                if audio_input_values.size(0) == 0:
                    print(f"Batch {i}: no valid samples after filtering, skipping.")
                    continue
                try:
                    _, fused_features = self(audio_input_values, audio_attention_mask, input_ids, attention_mask)
                except Exception as e:
                    print(f"Error during forward pass in init_prototypes for batch {i}: {e}")
                    break
                for feature, label in zip(fused_features, labels_batch):
                    class_features[label.item()].append(feature.cpu())
                del audio_input_values, audio_attention_mask, input_ids, attention_mask, labels_batch
                del fused_features
                torch.cuda.empty_cache()
        prototype_list = []
        for class_idx, class_embeds in enumerate(class_features):
            if not class_embeds:
                print(f"Warning: No samples found for class {class_idx}. Initializing prototypes randomly.")
                prototype_list.append(torch.randn(num_prototypes, self.metric_head.embed_dim))
                continue
            class_embeds = torch.stack(class_embeds)
            if len(class_embeds) < num_prototypes:
                print(f"Warning: Fewer samples ({len(class_embeds)}) than prototypes ({num_prototypes}) for class {class_idx}. Replicating samples.")
                indices = torch.randint(0, len(class_embeds), (num_prototypes,))
                proto_class = class_embeds[indices]
            else:
                indices = torch.randperm(len(class_embeds))[:num_prototypes]
                proto_class = class_embeds[indices]
            prototype_list.append(proto_class)
        while len(prototype_list) < num_classes:
            print(f"Warning: Missing prototypes for class {len(prototype_list)}. Initializing randomly.")
            prototype_list.append(torch.randn(num_prototypes, self.metric_head.embed_dim))
        proto_tensor = torch.stack(prototype_list).to(device)
        proto_tensor = proto_tensor.view(num_classes * num_prototypes, self.metric_head.embed_dim)
        self.metric_head.prototypes.data = proto_tensor
        print(f"Prototypes initialized: {self.metric_head.prototypes.shape}")

def compute_loss_weights(labels, num_classes, alpha=0.5, device='cpu'):
    label_count = torch.zeros(num_classes)
    for label in labels:
        label_count[label] += 1
    label_count = label_count.float()
    label_count_pow = label_count.pow(alpha)
    lw_weights = label_count_pow / label_count_pow.sum()
    lw_weights = lw_weights / (label_count + 1e-8)
    lw_weights = lw_weights.to(device)
    return lw_weights

def main():
    # Initialize processors
    global wav2vec_processor, tokenizer
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Wav2Vec2Processor and BertTokenizer initialized successfully.")

    # Data paths
    audio_base_dir = '/content/drive/MyDrive/ICNALE_SM_AUDIO'
    text_base_dir = '/content/drive/MyDrive/SM_0_Unclassified_Unmerged'
    audio_text_paths = []
    labels = []
    cefr_level_to_index = {
        'A2_0': 0,
        'B1_1': 1,
        'B1_2': 2,
        'B2_0': 3
    }
    text_file_map = {}
    for root, dirs, files in os.walk(text_base_dir):
        for file in files:
            if file.endswith('.txt'):
                base_filename = os.path.splitext(file)[0]
                parts = base_filename.split('_')
                if len(parts) >= 4:
                    region = parts[1]
                    student_id = parts[3]
                    text_file_map[(region, student_id)] = os.path.join(root, file)
                else:
                    print(f"Skipping text file due to unexpected filename format: {file}")
    audio_files = []
    for root, dirs, files in os.walk(audio_base_dir):
        for file in files:
            if file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    #audio_files = audio_files[:200]
    for audio_path in audio_files:
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            parts = base_filename.split('_')
            if len(parts) >= 5:
                region = parts[1]
                student_id = parts[3]
                if len(parts) >= 6 and parts[-2] in ['A2', 'B1', 'B2', 'C1']:
                    label_level = '_'.join(parts[-2:])
                else:
                    label_level = parts[-1]
                if (region, student_id) in text_file_map:
                    text_path = text_file_map[(region, student_id)]
                    if label_level in cefr_level_to_index:
                        label_index = cefr_level_to_index[label_level]
                        audio_text_paths.append((audio_path, text_path))
                        labels.append(label_index)
                    else:
                        print(f"Warning: Label '{label_level}' extracted from audio filename not found in cefr_level_to_index mapping for audio: {audio_path}")
                else:
                    print(f"Warning: Text file with Region '{region}' and Student ID '{student_id}' not found for audio: {audio_path}")
            else:
                print(f"Skipping audio file due to unexpected filename format (missing Region, Student ID or CEFR): {audio_path}")
        except IndexError:
            print(f"Skipping audio file due to unexpected filename format: {audio_path}")
    print(f"\nFound {len(audio_text_paths)} valid audio-text pairs.")

    audio_paths = [item[0] for item in audio_text_paths]
    text_paths = [item[1] for item in audio_text_paths]
    if len(audio_paths) > 0:
        train_audio_paths, temp_audio_paths, train_text_paths, temp_text_paths, train_labels, temp_labels = train_test_split(
            audio_paths, text_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        val_audio_paths, test_audio_paths, val_text_paths, test_text_paths, val_labels, test_labels = train_test_split(
            temp_audio_paths, temp_text_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        print(f"Training samples: {len(train_labels)}")
        print(f"Validation samples: {len(val_labels)}")
        print(f"Test samples: {len(test_labels)}")
        train_dataset = MultimodalDataset(train_audio_paths, train_text_paths, train_labels, wav2vec_processor, tokenizer)
        val_dataset = MultimodalDataset(val_audio_paths, val_text_paths, val_labels, wav2vec_processor, tokenizer)
        test_dataset = MultimodalDataset(test_audio_paths, test_text_paths, test_labels, wav2vec_processor, tokenizer)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        BATCH_SIZE = 8
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multimodal_collate_fn, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multimodal_collate_fn, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multimodal_collate_fn, num_workers=2)
        print(f"Number of batches in training dataloader: {len(train_dataloader)}")
        print(f"Number of batches in validation dataloader: {len(val_dataloader)}")
        print(f"Number of batches in test dataloader: {len(test_dataloader)}")
    else:
        print("No valid audio-text pairs found. Please check the audio path and file formats.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    NUM_CLASSES = len(cefr_level_to_index)
    K_PROTOTYPES = 3
    AUDIO_EMBED_DIM = 256
    TEXT_EMBED_DIM = 256
    COMBINED_EMBED_DIM = 512
    HIDDEN_DIM = 256
    model = MultimodalModel(
        num_classes=NUM_CLASSES,
        k=K_PROTOTYPES,
        audio_embed_dim=AUDIO_EMBED_DIM,
        text_embed_dim=TEXT_EMBED_DIM,
        combined_embed_dim=COMBINED_EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    print("Multimodal model instantiated successfully.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    if 'cefr_level_to_index' in locals() and len(train_labels) > 0:
        num_classes = len(cefr_level_to_index)
        lw_weights = compute_loss_weights(train_labels, num_classes=num_classes, alpha=0.5, device=device)
        criterion = nn.CrossEntropyLoss(weight=lw_weights)
        print("Loss function with class weights defined using cefr_level_to_index.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Loss function defined without class weights (cefr_level_to_index not found or train_labels empty).")
    if 'train_dataloader' in locals() and train_dataloader is not None:
        print("Initializing prototypes using training data.")
        model.init_prototypes(train_dataloader, num_classes=NUM_CLASSES, num_prototypes=K_PROTOTYPES)
    else:
        print("Warning: train_dataloader not found or is None. Prototypes not initialized with data.")
    print("Training setup complete.")

    EPOCHS = 15
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_labels = []
        print(f"Epoch {epoch+1}/{EPOCHS} - Training...")
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            audio_input_values = batch['audio_input_values'].to(device)
            audio_attention_mask = batch['audio_attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            optimizer.zero_grad()
            logits, _ = model(audio_input_values, audio_attention_mask, input_ids, attention_mask)
            batch_size = logits.shape[0]
            labels_batch = labels_batch[:batch_size]
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels_batch.cpu().numpy())
        avg_train_loss = total_train_loss / len(train_dataset)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds) * 100
        train_losses.append(avg_train_loss)
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []
        print(f"Epoch {epoch+1}/{EPOCHS} - Validating...")
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                audio_input_values = batch['audio_input_values'].to(device)
                audio_attention_mask = batch['audio_attention_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['label'].to(device)
                logits, _ = model(audio_input_values, audio_attention_mask, input_ids, attention_mask)
                batch_size = logits.shape[0]
                labels_batch = labels_batch[:batch_size]
                loss = criterion(logits, labels_batch)
                total_val_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels_batch.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_dataset)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds) * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Training Loss: {avg_train_loss:.4f} | Training Acc: {train_accuracy:.2f}%")
        print(f"  Validation Loss: {avg_val_loss:.4f} | Validation Acc: {val_accuracy:.2f}%\n")
    print("Training execution complete.")

if __name__ == "__main__":
    main()