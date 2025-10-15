#!/usr/bin/env python3
"""
Utility script to monitor and analyze memory usage during training.
"""

import torch
import argparse
from pathlib import Path
import re


def format_bytes(bytes_val):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def get_gpu_memory_info():
    """Get current GPU memory information"""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"GPU Memory Status ({num_gpus} GPU(s) detected)")
    print(f"{'='*60}\n")
    
    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        
        # Get memory info
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Calculate percentages
        alloc_pct = (allocated / total_memory) * 100
        reserved_pct = (reserved / total_memory) * 100
        max_alloc_pct = (max_allocated / total_memory) * 100
        
        print(f"GPU {i}: {torch.cuda.get_device_name(device)}")
        print(f"  Total Memory:     {format_bytes(total_memory)}")
        print(f"  Allocated:        {format_bytes(allocated)} ({alloc_pct:.1f}%)")
        print(f"  Reserved:         {format_bytes(reserved)} ({reserved_pct:.1f}%)")
        print(f"  Max Allocated:    {format_bytes(max_allocated)} ({max_alloc_pct:.1f}%)")
        print(f"  Free:             {format_bytes(total_memory - reserved)}")
        print()


def reset_peak_memory_stats():
    """Reset peak memory statistics"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        print("✓ Peak memory statistics reset")
    else:
        print("✗ CUDA not available")


def analyze_log_file(log_file):
    """Analyze memory usage from training log file"""
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing log file: {log_file}")
    print(f"{'='*60}\n")
    
    # Pattern to match memory logs
    pattern = r"\[Step (\d+)\] GPU Memory - Allocated: ([\d.]+)GB, Reserved: ([\d.]+)GB, Max: ([\d.]+)GB"
    
    memory_logs = []
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                allocated = float(match.group(2))
                reserved = float(match.group(3))
                max_alloc = float(match.group(4))
                memory_logs.append((step, allocated, reserved, max_alloc))
    
    if not memory_logs:
        print("No memory logs found in file")
        return
    
    # Calculate statistics
    allocated_values = [x[1] for x in memory_logs]
    reserved_values = [x[2] for x in memory_logs]
    max_values = [x[3] for x in memory_logs]
    
    print(f"Found {len(memory_logs)} memory snapshots")
    print(f"Step range: {memory_logs[0][0]} to {memory_logs[-1][0]}")
    print()
    print("Allocated Memory (GB):")
    print(f"  Min:     {min(allocated_values):.2f}")
    print(f"  Max:     {max(allocated_values):.2f}")
    print(f"  Average: {sum(allocated_values)/len(allocated_values):.2f}")
    print()
    print("Reserved Memory (GB):")
    print(f"  Min:     {min(reserved_values):.2f}")
    print(f"  Max:     {max(reserved_values):.2f}")
    print(f"  Average: {sum(reserved_values)/len(reserved_values):.2f}")
    print()
    print("Peak Allocated (GB):")
    print(f"  Max:     {max(max_values):.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GPU Memory Monitoring Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current GPU memory
  python tools/monitor.py
  
  # Reset peak memory statistics
  python tools/monitor.py --reset
  
  # Analyze memory from log file
  python tools/monitor.py --log logs/finetune-3155580.out
        """
    )
    
    parser.add_argument(
        '--reset', 
        action='store_true',
        help='Reset peak memory statistics'
    )
    parser.add_argument(
        '--log', 
        type=str,
        help='Analyze memory usage from log file'
    )
    
    args = parser.parse_args()
    
    if args.log:
        analyze_log_file(args.log)
    elif args.reset:
        reset_peak_memory_stats()
    else:
        get_gpu_memory_info()


if __name__ == "__main__":
    main()
