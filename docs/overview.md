# CULI Scoring Project - Comprehensive Overview

## Table of Contents
1. [Project Summary](#project-summary)
2. [Architecture Overview](#architecture-overview)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Configuration System](#configuration-system)
7. [Core Components](#core-components)
8. [Development Tools](#development-tools)
9. [Deployment & Infrastructure](#deployment--infrastructure)
10. [Getting Started](#getting-started)

---

## Project Summary

**CULI Scoring** is a multimodal deep learning system designed for **automatic language proficiency assessment** of English learners. The project leverages both **audio** and **text** modalities to provide accurate scoring and potentially feedback to language learners.

### Key Objectives
- 🎯 **Automatic Language Proficiency Scoring**: Classify language proficiency levels (A20, B11, B12, B20)
- 🔊 **Multimodal Analysis**: Combine audio speech patterns with textual content
- 📊 **High Accuracy**: Leverage state-of-the-art transformer models for feature extraction
- 🔄 **Scalable Training**: Support distributed training across multiple GPUs

### Target Dataset
- **ICNALE (International Corpus Network of Asian Learners of English)**
- Focus on **SM (Spoken Monologue)** subset
- 4-class classification: A20, B11, B12, B20 (CEFR proficiency levels)

### Core Design Principles
1. **Registry Pattern**: All components (models, data loaders, tasks) are registered for easy swapping
2. **Configuration-Driven**: Hydra-based configuration system for experiment management
3. **Modular Design**: Clear separation between data processing, model architecture, and training logic
4. **Type Safety**: Extensive use of Python protocols and type hints
5. **Distributed Training**: Built-in support for multi-GPU training with DDP

---

## Data Pipeline

### Data Flow Overview
```
Raw Audio Files → Feature Extraction → Artifact Storage → DataLoader → Model
      │                    │                │             │
      ├─ Wav2Vec2          ├─ _audio.pt     ├─ Batch      └─ Cross-Modal
      ├─ BERT Tokenizer    ├─ _token.pt     │   Creation      Fusion
      └─ Log-Mel Spec      └─ _logmel.pt    └─ Collation
```

### Feature Types & Artifacts

The system uses a **standardized artifact system** for consistent feature handling:

#### 1. **Token Artifacts** (`_token.pt`)
- **Source**: BERT tokenizer preprocessing
- **Content**: Dictionary with keys:
  - `input_ids`: Token IDs tensor
  - `attention_mask`: Attention mask tensor
  - `token_type_ids`: Segment IDs tensor
- **Usage**: Text understanding and semantic analysis

#### 2. **Encoded Artifacts** (`_audio.pt`)
- **Source**: Wav2Vec2 processor preprocessing
- **Content**: Dictionary with keys:
  - `input_values`: Normalized audio features
  - `attention_mask`: Optional attention mask
- **Usage**: Audio representation learning

#### 3. **LogMel Artifacts** (`_logmel.pt`)
- **Source**: Mel-spectrogram + AmplitudeToDB transformation
- **Content**: Direct tensor (no dictionary wrapper)
- **Shape**: `[1, n_mels, time_frames]`
- **Usage**: Traditional audio feature representation

### Data Loading Process

1. **CSV Metadata**: Contains file paths and labels
2. **Artifact Loading**: Uses type-safe artifact loaders
3. **Tensor Extraction**: Extracts specific tensors for model input:
   - `tokens`, `tokens_mask` (from TokenArtifact)
   - `encoded`, `encoded_mask` (from EncodedArtifact)
   - `logmel` (from LogMelArtifact)
4. **Batch Collation**: Handles variable-length sequences with padding

### K-Fold Cross-Validation
- Stratified group K-fold splitting
- Maintains label distribution across folds
- Separate train/validation sets per fold

---

## Model Architecture

### Cross-Modal Scorer Architecture

The core model (`CrossModalScorer`) implements a sophisticated multimodal fusion approach:

```python
# Key Components:
- Audio Encoder: Wav2Vec2Model (frozen/fine-tuned)
- Text Encoder: BertModel (frozen/fine-tuned)
- Cross-Attention: Multi-head attention between modalities
- Sequence Modeling: LSTM for temporal modeling
- Classification Head: Final scoring layer
```

#### Component Details

1. **Audio Encoding Path**
   - Input: Raw audio waveform
   - Wav2Vec2 → Audio representations (768-dim)
   - Positional encoding for sequence ordering

2. **Text Encoding Path**
   - Input: Tokenized text
   - BERT → Contextual word embeddings (768-dim)
   - Attention-based pooling

3. **Cross-Modal Fusion**
   - Cross-attention between audio and text features
   - Bidirectional information flow
   - Learned attention weights

4. **Sequence Modeling**
   - LSTM for capturing temporal dependencies
   - Hidden state: configurable (default: 512-dim)

5. **Classification**
   - Final linear layer
   - 4-class output (A20, B11, B12, B20)
   - Softmax activation for probability distribution

### Alternative Models

The project also includes **Phi-4** integration for advanced language modeling capabilities, showing the system's extensibility.

---

## Training Pipeline

### Training Flow
```
Configuration → Model Creation → Data Loading → Training Loop → Evaluation → Checkpointing
      │              │              │              │             │            │
   Hydra Configs → Registry → Artifact System → Loss Calc → Metrics → Model Save
```

### Key Training Components

#### 1. **Trainer Engine** (`src/engine/trainer.py`)
- Handles the complete training lifecycle
- Supports distributed training (DDP)
- Automatic mixed precision (AMP) support
- Gradient accumulation
- Learning rate scheduling

#### 2. **Task Definition** (`src/tasks/classification.py`)
- Defines loss computation (CrossEntropyLoss)
- Metrics calculation (Confusion Matrix)
- Training/validation step logic
- Result aggregation

#### 3. **Optimization**
- Adam optimizer with configurable parameters
- Learning rate scheduling
- Gradient clipping for stability

#### 4. **Monitoring & Logging**
- TensorBoard integration
- Metric tracking (accuracy, loss, confusion matrix)
- Model checkpointing
- Early stopping capabilities

### Distributed Training Support
- **Multi-GPU**: DDP (Distributed Data Parallel)
- **Infrastructure**: LANTA cluster with 4 GPUs
- **Synchronization**: Automatic gradient synchronization
- **Data Distribution**: Distributed sampling

---

## Configuration System

The project uses **Hydra** for sophisticated configuration management:

### Configuration Hierarchy
```
configs/
├── defaults.yaml              # Main configuration entry point
├── data/
│   └── icnale.yaml            # Data configuration
├── model/
│   └── icnale.yaml            # Model architecture config
├── task/
│   └── classification.yaml    # Task-specific settings
├── train/
│   └── base.yaml              # Training hyperparameters
├── eval/
│   └── base.yaml              # Evaluation settings
├── pipeline/
│   └── base.yaml              # Data processing pipeline
└── download/
    ├── base.yaml              # Download configurations
    └── phi4.yaml              # Model-specific downloads
```

### Configuration Features
- **Composition**: Mix and match different configs
- **Override**: Command-line parameter overrides
- **Experimentation**: Easy hyperparameter sweeps
- **Environment**: Different configs for different environments
- **Validation**: Type checking and validation

### Example Configuration Usage
```bash
# Basic training
python src/main.py

# Override data configuration
python src/main.py data=icnale data.features=[encoded,tokens,logmel]

# Change model architecture
python src/main.py model=icnale model.dropout=0.2

# Run evaluation only
python src/main.py cmd=eval
```

---

## Core Components

### 1. **Registry System** (`src/core/registry.py`)
- **Purpose**: Centralized component registration
- **Benefits**: Easy component swapping, plugin architecture
- **Usage**: Register models, data loaders, tasks, optimizers

```python
@register("model", "icnale")
def build_icnale_model(cfg):
    return CrossModalScorer(**cfg.model)

@register("data", "icnale")
def build_icnale_data(cfg):
    return ICNALEDataModule(cfg)
```

### 2. **Interface Protocols** (`src/interfaces/`)
- **BaseTask**: Task definition interface
- **DataModule**: Data loading interface
- **ModelModule**: Model interface
- **Artifact System**: Standardized feature loading

### 3. **Data Interfaces** (`src/interfaces/data.py`)
- **Sample**: Single data point container
- **Batch**: Batched data container
- **Type Safety**: Ensures consistent data flow

### 4. **Distributed Training** (`src/core/distributed.py`)
- **DDP Setup**: Automatic distributed training setup
- **Process Management**: Multi-process coordination
- **Synchronization**: Gradient and state synchronization

### 5. **I/O Operations** (`src/core/io.py`)
- **Checkpoint Management**: Model saving/loading
- **Artifact Handling**: Feature serialization
- **Path Management**: Consistent file handling

---

## Development Tools

### 1. **Code Analysis Tool** (`tools/tools.py`)
- **Snippet Extraction**: Extract functions, classes, imports
- **Code Overview**: Quick project understanding
- **Development Aid**: Reusable code discovery

```bash
# List available tools
python tools/tools.py --list

# Extract functions from specific tool
python tools/tools.py --tool eda --type functions

# Show overview of all tools
python tools/tools.py --all --type summary
```

### 2. **EDA Tools** (`tools/eda.py`)
- **Data Analysis**: CSV data exploration
- **Statistical Summary**: Data distribution analysis
- **Protocol-Based**: Extensible analyzer pattern

### 3. **Monitoring Tools** (`tools/monitor.py`)
- **File Monitoring**: Track file changes
- **Time-Based Filtering**: Monitor files by modification time
- **Extension Filtering**: Focus on specific file types

### 4. **File Management** (`tools/move.py`)
- **Safe File Operations**: Robust file/folder moving
- **Backup Protection**: Prevent data loss
- **Dry-Run Mode**: Preview operations before execution

---

## Deployment & Infrastructure

### Production Environment
- **Cluster**: LANTA supercomputer
- **GPU Configuration**: 4 x high-end GPUs
- **Training Mode**: Distributed Data Parallel (DDP)
- **Storage**: High-performance shared storage

### Model Serving
- **Checkpoint Format**: PyTorch state dictionaries
- **Model Registry**: Organized model versioning
- **Inference Pipeline**: Streamlined prediction workflow

### Scalability Features
- **Horizontal Scaling**: Multi-GPU training
- **Memory Optimization**: Gradient checkpointing
- **Efficient Data Loading**: Multi-worker data loaders
- **Batch Size Optimization**: Dynamic batch sizing

---

## Getting Started

### 1. **Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export PROJECT_ROOT=/path/to/project
```

### 2. **Data Preparation**
```bash
# Download raw data
python src/main.py cmd=download

# Process data pipeline
python src/main.py cmd=pipeline
```

### 3. **Training**
```bash
# Basic training
python src/main.py cmd=train

# Custom configuration
python src/main.py cmd=train data.features=[encoded,tokens] model.dropout=0.2

# Distributed training
python src/main.py cmd=train ddp=true
```

### 4. **Evaluation**
```bash
# Run evaluation
python src/main.py cmd=eval

# Custom evaluation dataset
python src/main.py cmd=eval data.test=path/to/test.csv
```

### 5. **Development Workflow**
```bash
# Analyze project structure
python tools/tools.py --all --type summary

# Monitor training progress
tensorboard --logdir outputs/

# Extract code snippets
python tools/tools.py --tool eda --type functions
```

---

## Project Status & Future Directions

### Current Status
- ✅ **Core Architecture**: Multimodal fusion pipeline implemented
- ✅ **Data Pipeline**: ICNALE dataset processing complete
- ✅ **Training Infrastructure**: DDP training on LANTA cluster
- ✅ **Artifact System**: Standardized feature handling
- 🔄 **In Progress**: Project reconstruction and optimization

### Future Enhancements
1. **Model Improvements**
   - Advanced attention mechanisms
   - Multi-scale audio features
   - Transformer-based fusion

2. **Data Expansion**
   - Additional language corpora
   - Cross-lingual proficiency assessment
   - Multi-task learning objectives

3. **Deployment**
   - Real-time inference API
   - Web-based assessment interface
   - Mobile application integration

4. **Feedback Generation**
   - Detailed proficiency analysis
   - Targeted improvement suggestions
   - Progress tracking over time

---

This overview provides a comprehensive understanding of the CULI Scoring project architecture, components, and development workflow. The modular design and configuration-driven approach make it easy to experiment with different models, datasets, and training strategies while maintaining code quality and reproducibility.