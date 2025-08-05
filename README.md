# CULI Scoring

## Baseline Review
* [Literature 1](https://aclanthology.org/2024.findings-naacl.86.pdf): **Wav2Vec2.0 -Mean Pooling-> MLP -> Prototyping Classifier -> Loss Reweighting**, reproduced with [train_baseline_1.py](https://github.com/tanntnny/culi-scoring/blob/main/scripts/train_baseline_1.py)
* All the training procedures are run in LANTA, using Distributed Data Parallel (DDP) with 4 GPUs.