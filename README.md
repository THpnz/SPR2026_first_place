# SPR2026 Mammography Report Classification: 1st place solution

## Overview
This repository provides a training pipeline for fine-tuning a BERT model on a multi-class text classification task.

The model uses a Portuguese pre-trained BERT and applies techniques to handle class imbalance effectively.

## Environment
- Docker image: nvcr.io/nvidia/pytorch:25.03-py3
- GPU: NVIDIA A100 (40GB)
- CPU: Intel Xeon Platinum 8275CL (3.00 GHz, 48 cores / 96 threads)
- RAM: 128GB

## Workflow

This project follows a two-step workflow:

1. **Train locally (GPU environment)**
2. **Run inference & submission on Kaggle**

---

## Step 1: Training (Local Environment)

Train the model locally to avoid Kaggle time limits.

### Run
`python train_bert.py`

### Outputs

After training, the following files will be generated:
```
results/exp001/  
├── bert_models/  
│   ├── bert_fold0.pth  
│   ├── bert_fold1.pth  
│   ├── ...  
├── submission_bert.csv
```

Tokenizer and config are also saved:
```
results/exp001/bert_models/  
├── tokenizer.json  
├── config.json  
```

## Step 2: Inference & Submission (Kaggle)

Use the trained models in the Kaggle Notebook:

🔗 https://www.kaggle.com/code/takayukihiguchi/notebook2c62e1a467

### Setup
1. Upload trained model files (bert_fold*.pth)
1. Upload tokenizer & config
1. Attach test dataset

Expected structure in Kaggle:
```
/kaggle/input/SPR 2026 Mammography Report Classification/
├── test.csv

/kaggle/input/your-models/
├── bert_fold0.pth
├── bert_fold1.pth
├── ...
├── tokenizer.json
├── config.json
```
