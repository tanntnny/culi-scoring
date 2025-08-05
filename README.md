# CULI Scoring

## Baseline Review
* [Literature 1](https://aclanthology.org/2024.findings-naacl.86.pdf): **Wav2Vec2.0 (mean pool) + MLP + SED + LW**, reproduced with [train_baseline_1.py](https://github.com/tanntnny/culi-scoring/blob/main/scripts/train_baseline_1.py)
* [Literature 1](https://aclanthology.org/2024.findings-naacl.86.pdf): **BERT Base (mean pool) + MLP + SED + LW**, reproduced with [train_baseline_2.py](https://github.com/tanntnny/culi-scoring/blob/main/scripts/train_baseline_1.py)
* All the training procedures are run in LANTA, using Distributed Data Parallel (DDP) with 4 GPUs.