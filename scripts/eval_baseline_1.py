import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import json

from scripts.utils.icnale_sm_audio_dataset import ICNALE_SM_Dataset, collate_fn
from scripts.utils.models import SpeechModel

def main():
    parser = argparse.ArgumentParser(description="Evaluate model and export predictions.")
    parser.add_argument("--val-data", type=str, required=True, help="Path to the ICNALE-SM validation dataset configuration.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save predictions JSON')
    parser.add_argument('--cefr-label', type=str, default='assets/cefr_label.csv', help='Path to CEFR label CSV')
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

    x_paths, y_true, y_pred = [], [], []
    total_batches = len(loader)
    with torch.no_grad():
        print("Starting evaluation...")
        for i, batch in enumerate(loader, 1):
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            paths = batch["paths"]
            logits = model(input_values, attention_mask)
            preds = logits.argmax(1).cpu().numpy()
            x_paths.extend(paths)
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())
            if i % 10 == 0 or i == total_batches:
                print(f"[Validation] Batch {i}/{total_batches} processed.")

    output = {
        "x_paths": [str(x) for x in x_paths],
        "y_true": [int(x) for x in y_true],
        "y_pred": [int(x) for x in y_pred]
    }
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(output, f)
    print(f"Predictions and true labels saved to {args.save_path}")

if __name__ == "__main__":
    main()
