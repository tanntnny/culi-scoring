#!/bin/bash
# Show diagnostic results from both stdout and stderr

if [ -z "$1" ]; then
    # Find most recent diagnostic job
    LATEST_OUT=$(ls -t logs/diagnose-*.out 2>/dev/null | head -1)
    if [ -z "$LATEST_OUT" ]; then
        echo "No diagnostic logs found."
        echo "Usage: $0 [JOB_ID]"
        exit 1
    fi
    JOB_ID=$(basename "$LATEST_OUT" | sed 's/diagnose-\(.*\)\.out/\1/')
else
    JOB_ID=$1
fi

OUT_FILE="logs/diagnose-${JOB_ID}.out"
ERR_FILE="logs/diagnose-${JOB_ID}.err"

echo "=========================================="
echo "Diagnostic Results for Job $JOB_ID"
echo "=========================================="
echo ""

if [ ! -f "$OUT_FILE" ]; then
    echo "Error: Output file not found: $OUT_FILE"
    exit 1
fi

echo "=== STDOUT (logs/diagnose-${JOB_ID}.out) ==="
echo ""
cat "$OUT_FILE"
echo ""

if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
    echo ""
    echo "=== STDERR (logs/diagnose-${JOB_ID}.err) ==="
    echo ""
    cat "$ERR_FILE"
    echo ""
fi

echo ""
echo "=========================================="

# Check for success
if grep -q "✅ ALL TESTS PASSED!" "$OUT_FILE"; then
    echo "Status: ✅ PASSED"
else
    echo "Status: ❌ FAILED"
    echo ""
    echo "To debug:"
    echo "  cat $OUT_FILE"
    echo "  cat $ERR_FILE"
fi

echo "=========================================="
