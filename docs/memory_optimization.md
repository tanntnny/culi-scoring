# Memory Optimization Guide for Phi4 Fine-tuning

## Applied Optimizations

### 1. Model-Level Optimizations

#### a) Gradient Checkpointing
- **Enabled**: `gradient_checkpointing: true`
- **Impact**: Reduces memory by ~40-50% at the cost of ~20% slower training
- **How it works**: Recomputes intermediate activations during backward pass instead of storing them

#### b) BFloat16 Precision
- **Changed**: `float16` → `bfloat16`
- **Impact**: Better numerical stability, same memory footprint as fp16
- **Benefit**: Reduces likelihood of gradient overflow/underflow

#### c) Disable KV Cache
- **Setting**: `use_cache: false`
- **Impact**: Saves memory during training (cache not needed for teacher forcing)
- **Note**: Only affects training; can be enabled for inference

#### d) Low CPU Memory Usage
- **Setting**: `low_cpu_mem_usage: true`
- **Impact**: Reduces CPU RAM usage during model loading by loading weights incrementally

### 2. Training-Level Optimizations

#### a) Automatic Mixed Precision (AMP)
- **Setting**: `amp: true`
- **Impact**: Uses fp16/bf16 for forward/backward, fp32 for optimizer
- **Benefit**: ~2x memory reduction, ~2-3x speed increase

#### b) Gradient Accumulation
- **Current**: `grad_accum: 1`
- **Recommendation**: Increase to 2-4 if you need larger effective batch size
- **How to adjust**:
  ```yaml
  grad_accum: 4  # Accumulate gradients over 4 steps
  ```

#### c) DDP Static Graph
- **Setting**: `ddp_static_graph: true`
- **Impact**: Reduces memory overhead in distributed training
- **Benefit**: Avoids re-registering hooks when using gradient checkpointing

### 3. Data-Level Optimizations

#### a) Reduce Batch Size
- **Current**: `batch: 1`
- **Status**: Already minimal
- **Note**: Use gradient accumulation for larger effective batch

#### b) DataLoader Workers
- **Current**: `num_workers: 8`
- **Recommendation**: Reduce if experiencing CPU memory pressure
  ```yaml
  num_workers: 4  # Or even 2
  ```

#### c) Pin Memory
- **Consider adding**: For faster CPU→GPU transfer
  ```python
  DataLoader(..., pin_memory=True)
  ```

## Expected Memory Savings

| Optimization | Memory Reduction | Speed Impact |
|-------------|------------------|--------------|
| Gradient Checkpointing | -40-50% | -20% |
| BFloat16 | 0% (vs fp16) | +5-10% |
| Disable KV Cache | -10-15% | 0% |
| AMP | -50% | +100-200% |
| **Total Estimated** | **~60-70%** | **+50-100%** |

## Monitoring Memory Usage

### During Training
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Profile Memory
Enable in config:
```yaml
profiler:
  profile_memory: true  # Currently false
```

## Additional Optimizations (If Needed)

### 1. Further Reduce Sequence Length
Trim audio/text to maximum required length:
```python
max_audio_length = 10 * 16000  # 10 seconds at 16kHz
```

### 2. Use DeepSpeed ZeRO
For multi-GPU setups, DeepSpeed ZeRO-2/3 can reduce memory significantly:
```yaml
deepspeed:
  zero_optimization:
    stage: 2
```

### 3. LoRA Rank Reduction
Reduce LoRA rank to decrease trainable parameters:
```python
# In model initialization
lora_config = LoraConfig(r=8)  # Reduce from default (likely 16)
```

### 4. Freeze More Layers
Only train the last N transformer blocks:
```python
for i, block in enumerate(model.model.layers):
    if i < len(model.model.layers) - 4:  # Freeze all but last 4
        for param in block.parameters():
            param.requires_grad = False
```

## Troubleshooting

### OOM During Forward Pass
- Reduce batch size further
- Enable gradient checkpointing
- Reduce sequence length

### OOM During Backward Pass
- Enable gradient checkpointing
- Reduce gradient accumulation steps
- Use optimizer state sharding (DeepSpeed)

### OOM During Optimizer Step
- Use 8-bit optimizers (bitsandbytes)
- Use CPU offloading for optimizer states

## Configuration Summary

**Current Optimized Config (`configs/model/phi4.yaml`):**
```yaml
name: phi4
src: models/phi4/phi4-model
torch_dtype: bfloat16
attn_implementation: flash_attention_2
low_cpu_mem_usage: true
use_cache: false
gradient_checkpointing: true
```

**Training Config (`configs/train/base.yaml`):**
```yaml
batch: 1
amp: true
grad_accum: 1  # Increase if needed
ddp_static_graph: true
clip_grad: 1.0
```
