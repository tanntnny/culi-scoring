import argparse
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
import json

from scripts.utils.icnale_sm_audio_dataset import ICNALE_SM_Dataset, collate_fn
from scripts.utils.models import SpeechModel

def main():
    parser = argparse.ArgumentParser(description="Evaluate model and export predictions.")
    parser.add_argument("--val-data", type=str, required=True, help="Path to the ICNALE-SM validation dataset configuration.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save predictions JSON')
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

    output = {
        "y_true": y_true,
        "y_pred": y_pred
    }
    with open(args.save_path, 'w') as f:
        json.dump(output, f)
    print(f"Predictions and true labels saved to {args.save_path}")

if __name__ == "__main__":
    main()
