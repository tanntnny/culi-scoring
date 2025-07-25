import os
from io import BytesIO
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Wav2Vec2Model, Wav2Vec2Processor, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
import torchaudio
import logging
from datetime import datetime

# ------------------- Utilities -------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("train_baseline_1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


logger.info("Loading Wav2Vec2 processor and CEFR labels.")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
cefr_label = pd.read_csv("assets/cefr_label.csv")

# Recursively find files in a folder tree
def dig_folder(file):
    returning = []
    if os.path.isdir(file):
        for f in os.listdir(file):
            returning.extend(dig_folder(os.path.join(file, f)))
    else:
        returning.append(file)
    return returning

# Convert MP3 to tensor
def mp3_to_tensor(mp3_path, frame_rate=16_000):
    logger.debug(f"Converting {mp3_path} to tensor.")
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(frame_rate).set_channels(1)
    buf = BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    waveform, sample_rate = torchaudio.load(buf)
    return waveform, sample_rate

# Create a data configuration DataFrame
def create_data_config(prefix):
    logger.info(f"Creating data config from {prefix}")
    paths = []
    labels = []
    for f in dig_folder(prefix):
        basename = os.path.basename(f)
        label = basename.split("_")[-2] + "_" + basename.split("_")[-1][0]
        if label in cefr_label["CEFR Level"].values:
            paths.append(f)
            labels.append(label)
    df = pd.DataFrame({
        'path': paths,
        'label': labels
    })
    logger.info(f"Found {len(df)} audio files for training/evaluation.")
    return df

# ------------------- Dataset -------------------

class ICNALE_SM_Dataset(Dataset):
    def __init__(self, data_config):
        logger.info(f"Initializing dataset with {len(data_config)} samples.")
        self.samples = [] # list of tuples (waveform, label)
        for _, row in data_config.iterrows():
            path, label = row['path'], row['label']
            try:
                waveform, _ = mp3_to_tensor(path)
                waveform = waveform.squeeze().numpy()
                value = cefr_label.loc[cefr_label["CEFR Level"] == label, "label"].values
                if len(value) > 0:
                    label = value[0]
                    self.samples.append((waveform, label))
                else:
                    logger.warning(f"Label '{label}' not found in CEFR label mapping for file: {path}")
            except Exception as e:
                logger.warning(f"Error processing file {path}: {e}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
    
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    proc_out = wav2vec_processor(
        waveforms,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    proc_out["labels"] = torch.tensor(labels, dtype=torch.long)
    return proc_out

# ------------------- Models -------------------

class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

class PrototypicalClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, k: int = 3):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(
            torch.randn(num_classes * k, embed_dim) / math.sqrt(embed_dim)
        )
        self.log_tau = nn.Parameter(torch.zeros(()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.prototypes, p=2) ** 2
        dists = dists.view(x.size(0), self.num_classes, self.k)
        dists = dists.mean(dim=2)
        logits = -dists / torch.exp(self.log_tau)
        return logits

class SpeechModel(nn.Module):
    def __init__(self, num_classes: int, k: int = 3):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_size = self.encoder.config.hidden_size
        self.pooler = MeanPooler()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.metric_head = PrototypicalClassifier(embed_dim=256, num_classes=num_classes, k=k)
    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        pooled = self.pooler(out.last_hidden_state, attention_mask)
        z = self.mlp(pooled)
        logits = self.metric_head(z)
        return logits

# ------------------- Train-Test Split -------------------

audio_path = "dataset/audio"
logger.info("Preparing train/test split.")
data_config = create_data_config(audio_path)
train_data_config, eval_data_config = train_test_split(
    data_config,
    test_size=0.2,
    random_state=42,
    stratify=data_config['label']
)
logger.info(f"Train samples: {len(train_data_config)}, Eval samples: {len(eval_data_config)}")

# ------------------- Distributed Training Setup -------------------

def setup_ddp():
    logger.info("Initializing distributed process group.")
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    logger.info(f"Process running on local_rank={local_rank}, device={torch.cuda.current_device()}")
    return local_rank

# ------------------- Training Loop -------------------

def run_epoch(model, loader, criterion, optimiser=None, scaler=None, device="cuda"):
    is_train = optimiser is not None
    model.train() if is_train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(batch["input_values"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
        if is_train:
            optimiser.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
        preds = logits.argmax(1)
        total_loss += loss.item() * preds.size(0)
        correct += (preds == batch["labels"]).sum().item()
        n += preds.size(0)
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}: Loss={loss.item():.4f}")
    return total_loss / n, correct / n
def save_model(model, epoch, eval_acc):
    save_path = f"checkpoints/model_epoch{epoch}_acc{eval_acc:.4f}.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

# ------------------- Main -------------------

def main():
    logger.info("Starting main training loop.")
    local_rank = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    NUM_CLASSES = len(cefr_label)
    K_PROTOTYPES = 3
    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 5e-5
    WARMUP_FRAC = 0.1

    train_dataset = ICNALE_SM_Dataset(train_data_config)
    eval_dataset = ICNALE_SM_Dataset(eval_data_config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = SpeechModel(num_classes=NUM_CLASSES, k=K_PROTOTYPES).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(EPOCHS * WARMUP_FRAC * len(train_loader)),
        num_training_steps=EPOCHS * len(train_loader)
    )

    best_eval_acc = 0.0
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS} started.")
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, scaler, device)
        eval_loss, eval_acc = run_epoch(model, eval_loader, criterion, None, None, device)
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc:.4f}")
        scheduler.step()
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            save_model(model, epoch+1, eval_acc)
            logger.info("New best model saved.")
        eval_loss, eval_acc = run_epoch(model, eval_loader, criterion, None, None, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")
        scheduler.step()

if __name__ == "__main__":
    main()
