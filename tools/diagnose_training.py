#!/usr/bin/env python3
"""
Diagnostic script to test the training setup and identify issues.
Run this before full training to catch errors early.

Can be run:
1. Locally: python tools/diagnose_training.py
2. On SLURM: sbatch scripts/diagnose.sh
3. Interactive SLURM: srun --gpus=1 --mem=32G python tools/diagnose_training.py
"""

import sys
import os
import traceback

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def print_environment_info():
    """Print environment information"""
    print("="*60)
    print("Environment Information")
    print("="*60)
    
    # Check if running on SLURM
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        print(f"Running on SLURM:")
        print(f"  Job ID: {slurm_job_id}")
        print(f"  Node: {os.environ.get('SLURM_NODELIST', 'N/A')}")
        print(f"  Tasks: {os.environ.get('SLURM_NTASKS', 'N/A')}")
        print(f"  GPUs: {os.environ.get('SLURM_GPUS_ON_NODE', 'N/A')}")
        print(f"  CPUs: {os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')}")
    else:
        print("Running locally (not on SLURM)")
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    print()


def test_imports():
    """Test that all required imports work"""
    print("="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        print("✓ Transformers")
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import hydra
        print("✓ Hydra")
    except Exception as e:
        print(f"✗ Hydra import failed: {e}")
        return False
    
    print()
    return True


def test_config_loading():
    """Test that config can be loaded"""
    print("="*60)
    print("Testing config loading...")
    print("="*60)
    
    try:
        from hydra import initialize, compose
        from omegaconf import OmegaConf
        
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="defaults", overrides=[
                "cmd=train",
                "data=phi4", 
                "model=phi4",
                "task=finetune_lora",
                "ddp=False",  # Test without DDP first
            ])
        
        print("✓ Config loaded successfully")
        print(f"  Model: {cfg.model.name}")
        print(f"  Data: {cfg.data.name}")
        print(f"  Task: {cfg.task.name}")
        print(f"  Batch size: {cfg.train.batch}")
        print(f"  AMP: {cfg.train.amp}")
        print()
        return cfg
    except Exception as e:
        print(f"✗ Config loading failed:")
        traceback.print_exc()
        return None


def test_model_loading(cfg):
    """Test that the model can be loaded"""
    print("="*60)
    print("Testing model loading...")
    print("="*60)
    
    try:
        from src.core.registry import build
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        
        print("  Loading model...")
        model = build("model", cfg.model.name, cfg=cfg)
        print(f"✓ Model loaded: {type(model).__name__}")
        
        # Try moving to device
        print(f"  Moving to {device}...")
        model = model.to(device)
        print(f"✓ Model on device")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        print()
        return model
    except Exception as e:
        print(f"✗ Model loading failed:")
        traceback.print_exc()
        return None


def test_data_loading(cfg):
    """Test that data can be loaded"""
    print("="*60)
    print("Testing data loading...")
    print("="*60)
    
    try:
        from src.core.registry import build
        
        print("  Loading datamodule...")
        datamodule = build("data", cfg.data.name, cfg=cfg)
        print(f"✓ Datamodule loaded: {type(datamodule).__name__}")
        
        print("  Getting train dataloader...")
        train_loader = datamodule.train_dataloader()
        print(f"✓ Train dataloader created")
        print(f"  Number of batches: {len(train_loader)}")
        
        # Try loading first batch
        print("  Loading first batch...")
        first_batch = next(iter(train_loader))
        print(f"✓ First batch loaded")
        
        if hasattr(first_batch, 'inputs'):
            print("  Batch inputs:")
            for key, val in first_batch.inputs.items():
                if hasattr(val, 'shape'):
                    print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"    {key}: {type(val)} = {val}")
        
        if hasattr(first_batch, 'outputs'):
            print("  Batch outputs:")
            for key, val in first_batch.outputs.items():
                if hasattr(val, 'shape'):
                    print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"    {key}: {type(val)} = {val}")
        
        print()
        return datamodule, first_batch
    except Exception as e:
        print(f"✗ Data loading failed:")
        traceback.print_exc()
        return None, None


def test_forward_pass(model, batch, cfg):
    """Test a forward pass through the model"""
    print("="*60)
    print("Testing forward pass...")
    print("="*60)
    
    try:
        import torch
        
        device = next(model.parameters()).device
        print(f"  Model device: {device}")
        
        # Move batch to device
        print("  Moving batch to device...")
        batch = batch.to(device)
        print("✓ Batch moved to device")
        
        # Try forward pass
        print("  Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(**batch.inputs)
        print(f"✓ Forward pass successful")
        print(f"  Output type: {type(output)}")
        if hasattr(output, 'shape'):
            print(f"  Output shape: {output.shape}")
        elif hasattr(output, 'logits'):
            print(f"  Output logits shape: {output.logits.shape}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Forward pass failed:")
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST FOR PHI4 TRAINING")
    print("="*60 + "\n")
    
    # Print environment info
    print_environment_info()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install required packages.")
        return False
    
    # Test config
    cfg = test_config_loading()
    if cfg is None:
        print("\n❌ Config test failed. Check your config files.")
        return False
    
    # Test model
    model = test_model_loading(cfg)
    if model is None:
        print("\n❌ Model test failed. Check model configuration.")
        return False
    
    # Test data
    datamodule, batch = test_data_loading(cfg)
    if datamodule is None or batch is None:
        print("\n❌ Data test failed. Check your data files and paths.")
        return False
    
    # Test forward pass
    if not test_forward_pass(model, batch, cfg):
        print("\n❌ Forward pass test failed. Check model and data compatibility.")
        return False
    
    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour setup appears to be working correctly.")
    print("You can now try running full training.\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
