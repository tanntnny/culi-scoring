#!/bin/bash
# Quick script to check diagnostic job results

echo "=========================================="
echo "Checking Diagnostic Results"
echo "=========================================="

# Find the most recent diagnostic job
LATEST_OUT=$(ls -t logs/diagnose-*.out 2>/dev/null | head -1)
LATEST_ERR=$(ls -t logs/diagnose-*.err 2>/dev/null | head -1)

if [ -z "$LATEST_OUT" ]; then
    echo "❌ No diagnostic logs found in logs/"
    echo ""
    echo "Have you run the diagnostic yet?"
    echo "Run: sbatch scripts/diagnose.sh"
    exit 1
fi

# Extract job ID
JOB_ID=$(basename "$LATEST_OUT" | sed 's/diagnose-\(.*\)\.out/\1/')

echo "Latest diagnostic job: $JOB_ID"
echo "Output file: $LATEST_OUT"
echo "Error file: $LATEST_ERR"
echo ""

# Check if job is still running
if squeue -j $JOB_ID &>/dev/null; then
    echo "⏳ Job is still running..."
    echo ""
    echo "Current output:"
    echo "------------------------------------------"
    tail -20 "$LATEST_OUT"
    echo "------------------------------------------"
    echo ""
    echo "Tip: Monitor with: tail -f $LATEST_OUT"
    exit 0
fi

# Job completed - check results
echo "📊 Job completed. Checking results..."
echo ""

# Check for success marker
if grep -q "✅ ALL TESTS PASSED!" "$LATEST_OUT"; then
    echo "=========================================="
    echo "✅ DIAGNOSTIC PASSED!"
    echo "=========================================="
    echo ""
    echo "Your setup is working correctly."
    echo "You can now submit your training job:"
    echo "  sbatch scripts/experiment/finetune_lora.sh"
    echo ""
    
    # Show summary
    echo "Test Results:"
    grep -E "(✓|✗)" "$LATEST_OUT" | head -10
    
    exit 0
else
    echo "=========================================="
    echo "❌ DIAGNOSTIC FAILED"
    echo "=========================================="
    echo ""
    echo "Last 30 lines of output:"
    echo "------------------------------------------"
    tail -30 "$LATEST_OUT"
    echo "------------------------------------------"
    echo ""
    
    # Check error log
    if [ -s "$LATEST_ERR" ]; then
        echo "Errors found in $LATEST_ERR:"
        echo "------------------------------------------"
        cat "$LATEST_ERR"
        echo "------------------------------------------"
        echo ""
    fi
    
    echo "What to do:"
    echo "1. Read the error message above"
    echo "2. Fix the issue (see docs/ddp_error_fix.md)"
    echo "3. Run diagnostic again: sbatch scripts/diagnose.sh"
    echo ""
    echo "View full output: cat $LATEST_OUT"
    
    exit 1
fi
