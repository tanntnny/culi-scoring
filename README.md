# CULI Scoring
This repository contains code and resources for the CULI Scoring project, which aims to improve the automatic scoring of language proficiency and potentially give feedback to learners.

## Baseline Review
* [Literature 1](https://aclanthology.org/2024.findings-naacl.86.pdf): **Wav2Vec2.0 (Mean Pool) + PT(SED) + LW**, reproduced with [train_baseline_1.py](https://github.com/tanntnny/culi-scoring/blob/main/scripts/train_baseline_1.py)

## Proposed Solutions
* Pipeline 2: **Wav2Vec2.0 (BiLSTM) + BERT (BiLSTM) + PT(SED)**, produced with [train_pipeline_2.py](https://github.com/tanntnny/culi-scoring/blob/main/scripts/train_pipeline_2.py)

## More Information
* All the training procedures are run in **LANTA**, using Distributed Data Parallel (DDP) with 4 GPUs.