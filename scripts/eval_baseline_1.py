import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from scripts.utils.icnale_sm_audio_dataset import ICNALE_SM_Dataset, collate_fn
from scripts.utils.models import SpeechModel

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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load label mapping and processor
    cefr_label = pd.read_csv(args.cefr_label)
    val_config = pd.read_csv(args.val_data)

    # Load model
    num_classes = len(cefr_label)
    model = SpeechModel(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Prepare dataset and loader
    dataset = ICNALE_SM_Dataset(val_config, cefr_label)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_values, attention_mask)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    # Save confusion matrix
    plot_and_save_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), save_path=args.save_path)
    print(f"Confusion matrix saved to {args.save_path}")

if __name__ == "__main__":
    main()
