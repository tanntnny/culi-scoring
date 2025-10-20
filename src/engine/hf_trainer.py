from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import numpy as np
from transformers import EvalPrediction
from transformers import (
    Trainer as HFTrainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_scheduler
)
from transformers.trainer_utils import get_last_checkpoint
from ..core.registry import build, register
from ..core.logging import Logger
from ..core.distributed import init_distributed_if_needed, is_global_zero, cleanup_distributed, is_dist, barrier
from ..core.profiler import TrainingProfiler

class HuggingFaceTrainer:
    """
    Wrapper around HuggingFace Trainer to integrate with the existing codebase structure.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        init_distributed_if_needed(cfg.ddp)
        self._use_ddp = cfg.ddp and is_dist()
        
        # Warn if DDP was requested but not initialized
        if cfg.ddp and not self._use_ddp:
            if is_global_zero():
                print("[HFTrainer] Warning: DDP was requested but distributed process group is not initialized. Running without DDP.")

        # Set device
        if self._use_ddp and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.datamodule = build("data", cfg.data.name, cfg=cfg)
        self.model = build("model", cfg.model.name, cfg=cfg).to(self.device)
        self.task = build("task", cfg.task.name, cfg=cfg)
        self.task.setup(self.model)
        
        # Setup trainer
        self._setup_trainer()
        
        # Logger for distributed training
        self.logger = Logger(Path(self.cfg.output_dir) / "tb") if is_global_zero() else None

    def _move_to_device(self, obj, device):
        """Recursively move tensors in a nested structure to the given device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._move_to_device(v, device) for v in obj]
            return type(obj)(t) if isinstance(obj, tuple) else t
        return obj

    def _pre_training_profile(self):
        """Run a short profiling warm-up before the main training loop."""
        # Guard clauses for config presence
        prof_cfg = getattr(self.cfg.train, "profiler", None)
        if prof_cfg is None:
            return
        enabled = bool(prof_cfg.get("enabled", True))
        pre_steps = int(prof_cfg.get("pre_steps", 0))
        if (not enabled) or pre_steps <= 0:
            return

        if is_global_zero():
            print(f"[HFTrainer] Running pre-training profiler for {pre_steps} step(s)...")

        do_backward = bool(prof_cfg.get("pre_backward", True))
        profiler = TrainingProfiler(prof_cfg, logger=self.logger)

        # Use the datamodule's train dataloader for realistic batches
        dl = self.datamodule.train_dataloader()
        self.model.train()
        # Enable gradient checkpointing during warm-up if configured
        try:
            if bool(getattr(self.cfg.train.hf_trainer, "gradient_checkpointing", False)):
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
        except Exception:
            pass

        step = 0
        profiler.start_epoch(epoch_index=0, global_step_start=0)
        try:
            use_fp16 = bool(getattr(self.cfg.train.hf_trainer, "fp16", False))
            use_bf16 = bool(getattr(self.cfg.train.hf_trainer, "bf16", False))
            use_amp = (use_fp16 or use_bf16) and (self.device.type == "cuda")
            amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)

            for batch in dl:
                batch = self._move_to_device(batch, self.device)
                if use_amp and amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = self.model(**batch)
                        loss = getattr(outputs, "loss", None)
                else:
                    outputs = self.model(**batch)
                    loss = getattr(outputs, "loss", None)

                if do_backward and loss is not None:
                    if use_amp and amp_dtype is not None:
                        loss.backward()
                    else:
                        loss.backward()
                    # Clear grads to avoid accumulation across warm-up
                    if hasattr(self.model, "zero_grad"):
                        self.model.zero_grad(set_to_none=True)

                step += 1
                profiler.step()
                if step >= pre_steps:
                    break
        except Exception as e:
            if is_global_zero():
                print(f"[HFTrainer] Pre-training profiler encountered an error: {e}")
            # Still attempt to properly close/stop the profiler
            profiler.stop_epoch(epoch_index=0, global_step_end=step, error=e)
            profiler.close()
            raise
        else:
            profiler.stop_epoch(epoch_index=0, global_step_end=step)
            profiler.close()

        # Sync ranks before proceeding to main training
        barrier()

    def _setup_trainer(self):
        """Setup HuggingFace Trainer with configuration."""
        # Get HF trainer config
        hf_config = self.cfg.train.hf_trainer
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=hf_config.output_dir,
            overwrite_output_dir=hf_config.overwrite_output_dir,
            
            # Training parameters
            per_device_train_batch_size=hf_config.per_device_train_batch_size,
            per_device_eval_batch_size=hf_config.per_device_eval_batch_size,
            gradient_accumulation_steps=hf_config.gradient_accumulation_steps,

            # Learning rate and scheduling
            learning_rate=hf_config.learning_rate,
            weight_decay=hf_config.weight_decay,
            warmup_steps=hf_config.warmup_steps,
            lr_scheduler_type=hf_config.lr_scheduler_type,
            
            # Training duration
            num_train_epochs=hf_config.num_train_epochs,
            max_steps=hf_config.max_steps,
            
            # Optimization
            fp16=hf_config.fp16,
            bf16=hf_config.bf16,
            tf32=hf_config.tf32,
            gradient_checkpointing=hf_config.gradient_checkpointing,
            gradient_checkpointing_kwargs=dict(hf_config.get("gradient_checkpointing_kwargs", {})),
            
            # Gradient clipping
            max_grad_norm=hf_config.max_grad_norm,
            
            # Evaluation
            evaluation_strategy=hf_config.evaluation_strategy,
            eval_steps=hf_config.eval_steps,
            eval_accumulation_steps=hf_config.eval_accumulation_steps,
            
            # Logging
            logging_strategy=hf_config.logging_strategy,
            logging_steps=hf_config.logging_steps,
            logging_first_step=hf_config.logging_first_step,
            disable_tqdm=hf_config.disable_tqdm,
            
            # Saving
            save_strategy=hf_config.save_strategy,
            save_steps=hf_config.save_steps,
            save_total_limit=hf_config.save_total_limit,
            load_best_model_at_end=hf_config.load_best_model_at_end,
            metric_for_best_model=hf_config.metric_for_best_model,
            greater_is_better=hf_config.greater_is_better,
            
            # Reproducibility
            seed=hf_config.seed,
            data_seed=hf_config.data_seed,
            
            # Data loading
            dataloader_num_workers=hf_config.dataloader_num_workers,
            dataloader_pin_memory=hf_config.dataloader_pin_memory,
            dataloader_persistent_workers=hf_config.dataloader_persistent_workers,
            
            # Distributed training
            ddp_find_unused_parameters=hf_config.ddp_find_unused_parameters,
            ddp_bucket_cap_mb=hf_config.ddp_bucket_cap_mb,
            ddp_broadcast_buffers=hf_config.ddp_broadcast_buffers,
            
            # Memory optimization
            dataloader_drop_last=hf_config.dataloader_drop_last,
            remove_unused_columns=hf_config.remove_unused_columns,
            
            # Reporting
            report_to=hf_config.report_to,
            logging_dir=hf_config.logging_dir,
            
            # Advanced settings
            push_to_hub=hf_config.push_to_hub,
            resume_from_checkpoint=hf_config.resume_from_checkpoint,
            ignore_data_skip=hf_config.ignore_data_skip,
            
            # Model saving format
            save_safetensors=hf_config.save_safetensors,
            
            # Deepspeed Config
            deepspeed=(hf_config.deepspeed if hf_config.deepspeed_enabled else None),
        )        
        # Get datasets
        train_dataset = self.datamodule.train_dataloader().dataset
        eval_dataset = None
        if hasattr(self.datamodule, 'val_dataloader') and self.datamodule.val_dataloader() is not None:
            eval_dataset = self.datamodule.val_dataloader().dataset
        
        # Setup callbacks
        callbacks = []
        if hf_config.get("early_stopping_patience"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=hf_config.early_stopping_patience,
                    early_stopping_threshold=hf_config.get("early_stopping_threshold", 0.0)
                )
            )
        
        # Get collate function from datamodule
        data_collator = None
        if hasattr(self.datamodule, "collator"):
            data_collator = self.datamodule.collator
        elif hasattr(self.datamodule, 'collate_fn'):
            data_collator = self.datamodule.collate_fn
        elif hasattr(self.datamodule.train_dataloader(), 'collate_fn'):
            data_collator = self.datamodule.train_dataloader().collate_fn

        

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            labels = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
            accuracy = (preds == labels).astype(np.float32).mean().item()
            return {"accuracy": accuracy}

        # Create HF Trainer
        self.trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        
        # Apply task-specific setup to trainer if needed
        if hasattr(self.task, 'setup_trainer'):
            self.task.setup_trainer(self.trainer)

    def fit(self):
        """Train the model using HuggingFace Trainer."""
        try:
            if is_global_zero():
                print("[HFTrainer] Starting training...")
                
            if self.cfg.get("profiler", False) and self.cfg.profiler.get("enabled", False):
                self._pre_training_profile()

            # Check for checkpoints
            resume_from_checkpoint = None
            if self.cfg.train.hf_trainer.get("resume_from_checkpoint"):
                resume_from_checkpoint = self.cfg.train.hf_trainer.resume_from_checkpoint
            else:
                # Auto-detect last checkpoint
                last_checkpoint = get_last_checkpoint(self.cfg.train.hf_trainer.output_dir)
                if last_checkpoint:
                    resume_from_checkpoint = last_checkpoint
                    if is_global_zero():
                        print(f"[HFTrainer] Resuming from checkpoint: {last_checkpoint}")
            
            # Start training
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save final model
            self.trainer.save_model()
            
            # Log training results
            if is_global_zero() and self.logger:
                self.logger.info(f"Training completed. Final train loss: {train_result.training_loss:.4f}")
                
            # Run final evaluation if eval dataset exists
            if self.trainer.eval_dataset is not None:
                eval_result = self.trainer.evaluate()
                if is_global_zero() and self.logger:
                    self.logger.info(f"Final evaluation results: {eval_result}")
                    
        except Exception as e:
            if is_global_zero():
                print(f"[HFTrainer] Training failed with error: {e}")
            raise
        finally:
            # Cleanup distributed training
            cleanup_distributed()
            if is_global_zero():
                print("[HFTrainer] Training session ended.")

    def evaluate(self):
        """Evaluate the model using HuggingFace Trainer."""
        if self.trainer.eval_dataset is None:
            if is_global_zero():
                print("[HFTrainer] No evaluation dataset available.")
            return None
            
        if is_global_zero():
            print("[HFTrainer] Starting evaluation...")
            
        eval_result = self.trainer.evaluate()
        
        if is_global_zero() and self.logger:
            self.logger.info(f"Evaluation results: {eval_result}")
            
        return eval_result

    def predict(self, dataset=None):
        """Make predictions using HuggingFace Trainer."""
        if dataset is None:
            dataset = self.trainer.eval_dataset
            
        if dataset is None:
            if is_global_zero():
                print("[HFTrainer] No dataset provided for prediction.")
            return None
            
        if is_global_zero():
            print("[HFTrainer] Starting prediction...")
            
        predictions = self.trainer.predict(dataset)
        
        if is_global_zero() and self.logger:
            self.logger.info(f"Prediction completed. Predictions shape: {predictions.predictions.shape}")
            
        return predictions