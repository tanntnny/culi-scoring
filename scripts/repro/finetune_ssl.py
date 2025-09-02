import argparse
from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    HubertModel,
    Wav2Vec2Processor
)

from scripts.utils.pytorch_utils import (
    initialize_distributed_training,
)

from scripts.data.multimodal_dataset import (
    MultimodalSMDataset
)

def finetune_ctc(args):
    # Setup DDP
    world_size, rank, local_rank, gpus_per_node = initialize_distributed_training()
    is_main = rank == 0
    num_workers = args.cpus
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    #
    if is_main:
        print(f"------------------- Arguments -------------------")
        print(f"Script: finetune_ssl.py (finetune_ctc)")
        print(f"Model: {args.model}")
        print(f"Train data: {args.train_data}")
        print(f"Val data: {args.val_data}")
        print(f"Seed: {args.seed}")

    
    # ---------------- Data --------------------
    train_data = MultimodalSMDataset(args.train_data)
    val_data = MultimodalSMDataset(args.val_data)    
    collate_fn = MultimodalSMDataset.create_collate_fn(
        audio_processor=args.audio_processor,
        text_tokenizer=args.text_tokenizer
    )

    train_sampler = DistributedSampler(
        train_data,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_data,
        shuffle=False,
        num_replicas=world_size,
        rank=rank,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---------------- Model, Criterion, Optimizer --------------------
    audio_processor = Wav2Vec2Processor.from_pretrained(args.processor)

    if args.model.startswith("facebook/hubert"):
        model = HubertModel.from_pretrained(
            args.model,
            ctc_loss_reduction="mean",
            pad_token_id=audio_processor.tokenizer.pad_token_id,
            vocab_size=len(audio_processor.tokenizer),
        )
    
    model.to(device)
    model.freeze_feature_extractor()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameers(), lr=args.learning_rate)

    if is_main:
        print(f"------------------- Model details & Devices -------------------")
        print(f"DDP initialized with device {device}") 
        print(f"Model architecture:\n{model.module}")
        num_params = sum(p.numel() for p in model.module.parameters())
        num_trainable = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,} | Trainable: {num_trainable:,}")
        print(f"Train samples: {len(train_data)} | Validation samples: {len(val_data)}")


    for epoch in range(args.epochs):
        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs}")

        model.train()
        for batch_idx, batch in enumerate(train_data):
            batch.to(device)
            logits = model(**batch["inputs"])
            with torch.no_grad():
                model.encoder._get_feat_extract_outputs_lengths


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a self-supervised model.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ctc = sub.add_parser("finetune_ctc", help="Fine-tune with CTC loss.")
    p_ctc.add_argument("--model", type=Union[str, Path], required=True, help="Path to load the model.")
    p_ctc.add_argument("--processor", type=Union[str, Path], required=True, help="Path to load the audio processor.")
    p_ctc.add_argument("--train-data", type=Union[str, Path], required=True, help="Path to training data.")
    p_ctc.add_argument("--val-data", type=Union[str, Path], required=True, help="Path to validation data.")
    p_ctc.add_argument("--batch-size", default=8, type=int, help="Batch size per GPU.")
    p_ctc.add_argument("--epochs", default=10, type=int, help="Number of training epochs.")
    p_ctc.add_argument("--learning-rate", default=1e-4, type=float, help="Learning rate.")
    p_ctc.add_argument("--seed", default=42, type=int, help="Random seed for initialization.")

    args = parser.parse_args()

    if args.cmd == "finetune_ctc":
        finetune_ctc(args)

if __name__ == "__main__":
    main()