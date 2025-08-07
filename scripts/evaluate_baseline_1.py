import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Dummy model and dataset classes for illustration; replace with your actual ones
class ICNALE_SM_Dataset(Dataset):
    def __init__(self, data_config, cefr_label_df, wav2vec_processor):
        self.cefr_label_df = cefr_label_df
        self.wav2vec_processor = wav2vec_processor
        self.samples = []
        for _, row in data_config.iterrows():
            path, label = row['path'], row['label']
            value = cefr_label_df.loc[cefr_label_df["CEFR Level"] == label, "label"].values
            if len(value) > 0:
                self.samples.append((path, int(value[0])))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, _ = torchaudio.load(path)
        return waveform.squeeze().numpy(), label

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    proc_out = wav2vec_processor(
        waveforms,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    if "attention_mask" not in proc_out:
        input_values = proc_out["input_values"]
        proc_out["attention_mask"] = torch.ones(input_values.shape, dtype=torch.long)
    proc_out["labels"] = torch.tensor(labels, dtype=torch.long)
    return proc_out

def plot_and_save_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate model and save confusion matrix.")
    parser.add_argument("--val-data", type=str, required=True, help="Path to the ICNALE-SM validation dataset configuration.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save confusion matrix image')
    parser.add_argument('--cefr_label', type=str, default='assets/cefr_label.csv', help='Path to CEFR label CSV')
    parser.add_argument('--processor_path', type=str, default='models/wav2vec2-processor', help='Path to Wav2Vec2 processor')
    args = parser.parse_args()

    # Load label mapping and processor
    cefr_label = pd.read_csv(args.cefr_label)
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.processor_path)
    data_config = pd.read_csv(args.data_config)

    # Load model (replace with your actual model class)
    num_classes = len(cefr_label)
    model = SpeechModel(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    # Prepare dataset and loader
    dataset = ICNALE_SM_Dataset(data_config, cefr_label, wav2vec_processor)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_values"], batch["attention_mask"])
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["labels"].cpu().numpy())

    # Save confusion matrix
    plot_and_save_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), save_path=args.save_path)
    print(f"Confusion matrix saved to {args.save_path}")

if __name__ == "__main__":
    main()
