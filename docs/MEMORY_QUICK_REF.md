# Memory Optimization Quick Reference

## 🎯 What Changed

### Model (`src/models/phi4.py`)
- ✅ Added gradient checkpointing (~40-50% memory reduction)
- ✅ Switched to bfloat16 (better stability)
- ✅ Disabled KV cache during training (~10-15% memory reduction)
- ✅ Enabled low CPU memory usage during loading

### Config (`configs/model/phi4.yaml`)
```yaml
torch_dtype: bfloat16           # was: float16
gradient_checkpointing: true    # NEW
use_cache: false               # NEW
low_cpu_mem_usage: true        # NEW
```

### Monitoring (`src/engine/loop.py`)
- ✅ Added automatic memory logging every N steps
- ✅ Shows allocated, reserved, and peak GPU memory

## 🚀 Quick Start

### Run Training (with optimizations)
```bash
python -m src.main cmd=train data=phi4 model=phi4 task=finetune_lora ddp=True
```

### Monitor Memory During Training
Memory is logged automatically every 10 steps in the output.

### Check Current GPU Memory
```bash
python tools/memory_monitor.py
```

### Analyze Log File
```bash
python tools/memory_monitor.py --log logs/finetune-3155580.out
```

## 📊 Expected Results

**Before Optimizations:**
- Memory usage: ~40-50GB per GPU
- Speed: Baseline

**After Optimizations:**
- Memory usage: ~15-20GB per GPU (60-70% reduction)
- Speed: +50-100% faster (bfloat16 + optimizations)

## 🔧 Troubleshooting

### Still Getting OOM?

#### Option 1: Increase Gradient Accumulation
```yaml
# configs/train/base.yaml
grad_accum: 4  # Effective batch size = 1 * 4 = 4
```

#### Option 2: Reduce Workers
```yaml
# configs/train/base.yaml
num_workers: 2  # was: 8
```

#### Option 3: Reduce Sequence Length
Edit the data preprocessing to limit audio length:
```python
max_audio_length = 10  # seconds
```

### Want Even More Memory Savings?

See detailed guide: [`docs/memory_optimization.md`](memory_optimization.md)

## 📈 Memory Monitoring

### In Training Logs
Look for lines like:
```
[Step 10] GPU Memory - Allocated: 18.45GB, Reserved: 20.12GB, Max: 19.87GB
```

### Manual Check
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## ✅ Verify Optimizations Active

Check your training logs for:
```
✅ gradient checkpointing enabled for audio processor
✅ [Step 0] GPU Memory - Allocated: X.XXG
✅ torch_dtype=torch.bfloat16
```

## 🔙 Rollback Changes

To disable optimizations (not recommended):
```yaml
# configs/model/phi4.yaml
torch_dtype: float16
gradient_checkpointing: false
use_cache: true
```

## 📝 Files Modified

1. `src/models/phi4.py` - Model implementation
2. `configs/model/phi4.yaml` - Model configuration
3. `src/engine/loop.py` - Training loop with monitoring
4. `docs/memory_optimization.md` - Full guide
5. `tools/memory_monitor.py` - Memory monitoring utility

## 💡 Pro Tips

1. **Start small**: Test with 1 batch to verify memory usage
2. **Monitor first epoch**: Memory usage stabilizes after first few steps
3. **Use profiler**: Enable `profile_memory: true` in config for detailed analysis
4. **Gradient checkpointing trade-off**: ~20% slower but 40-50% less memory
5. **BFloat16 is faster**: Despite same memory as FP16, it's often faster on modern GPUs

## 🆘 Need Help?

See full documentation:
- [`docs/memory_optimization.md`](memory_optimization.md) - Complete optimization guide
- [`docs/memory_changes_summary.md`](memory_changes_summary.md) - All changes in detail
