#!/bin/bash
# Quick diagnostic helper to identify model loading issues

echo "=========================================="
echo "Model Loading Issue Debugger"
echo "=========================================="
echo ""

cd /lustrefs/disk/project/pv823002-ulearn/culi 2>/dev/null || {
    echo "Error: Cannot cd to project directory"
    echo "Run this from: /lustrefs/disk/project/pv823002-ulearn/culi"
    exit 1
}

echo "1. Checking config..."
if [ -f "configs/model/phi4.yaml" ]; then
    echo "✓ Config exists"
    echo ""
    echo "Config contents:"
    cat configs/model/phi4.yaml
    echo ""
    
    # Extract model path
    MODEL_SRC=$(grep "^src:" configs/model/phi4.yaml | awk '{print $2}')
    echo "Model source path: $MODEL_SRC"
    echo ""
else
    echo "✗ Config not found: configs/model/phi4.yaml"
    exit 1
fi

echo "2. Checking model path..."
if [ -d "$MODEL_SRC" ]; then
    echo "✓ Model directory exists: $MODEL_SRC"
    echo ""
    echo "Contents:"
    ls -lh "$MODEL_SRC" | head -20
    echo ""
    
    # Check for required files
    REQUIRED_FILES=("config.json")
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$MODEL_SRC/$file" ]; then
            echo "  ✓ $file"
        else
            echo "  ✗ $file (missing)"
        fi
    done
    echo ""
else
    echo "✗ Model directory NOT found: $MODEL_SRC"
    echo ""
    echo "Checking what exists in models/:"
    ls -la models/ 2>/dev/null || echo "  models/ directory doesn't exist"
    echo ""
    echo "💡 Possible fixes:"
    echo "   1. Download the model first"
    echo "   2. Update config to use HuggingFace repo directly"
    echo "   3. Check if model is in a different location"
    echo ""
fi

echo "3. Checking latest diagnostic output..."
LATEST_ERR=$(ls -t logs/diagnose-*.err 2>/dev/null | head -1)
if [ -n "$LATEST_ERR" ] && [ -s "$LATEST_ERR" ]; then
    echo "Found error log: $LATEST_ERR"
    echo ""
    echo "Error contents:"
    echo "----------------------------------------"
    cat "$LATEST_ERR"
    echo "----------------------------------------"
    echo ""
else
    echo "No error log found or empty"
    echo ""
fi

echo "4. Checking Python environment..."
python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except:
    print('PyTorch not available')
try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except:
    print('Transformers not available')
" 2>&1

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

if [ -d "$MODEL_SRC" ]; then
    echo "Model path exists: ✓"
    echo ""
    echo "Next step: Run diagnostic again"
    echo "  sbatch scripts/diagnose.sh"
else
    echo "Model path missing: ✗"
    echo ""
    echo "Next steps:"
    echo "  1. View full error: bash scripts/show_diagnostic.sh"
    echo "  2. Download model or update config"
    echo "  3. See: docs/model_loading_error.md"
fi

echo ""
