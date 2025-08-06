import os
import torch

# Training Configuration
TRAIN_MODE = 'train'  # 'train' or 'inference'
SAVE_RESULT = False

# Dataset Configuration
DATASET_ROOT = '../Total_Datasets/Micro_Video_Rec_Dataset'
DATASET_NAME_LIST = ['KuaiRand-1K-Rand']
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

# Model Configuration
MODEL_NAMES = ['DPSkipNet']
MODEL_DIR = 'models'
EMBED_DIM = 128

# Training Parameters
ITERATIONS = [1, 1]  # [start, end]
EPOCHS = 30
BATCH_SIZE = 1024
NUM_WORKERS = 16
EARLY_STOP = 3

# Optimizer Configuration
OPTIMIZER = 'AdamW'
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Learning Rate Scheduler Configuration
LR_SCHEDULER = 'CosineAnnealingLR'
T_MAX = EPOCHS
T_0 = EPOCHS
ETA_MIN = 1e-6

# Device Configuration
DEVICES = [0]

# Evaluation Configuration
TOP_KS = [3, 5, 10, 20]
VAL_CRITERION_TOP_K = 1

# Output Configuration
OUTPUT_ROOT = 'output'
GRAPH_ROOT = 'graphs'

# Inference Configuration (for inference mode)
PAST_OUTPUT_ROOT = 'output'
PAST_RESULT_CSV = 'output/Mircro_Video_Recommendation_250522_021058.csv'

# Metrics Configuration
BASE_METRICS = ['Experient Time', 'Train Time', 'Dataset Name', 'Model Name', 'Iteration', 'Val Recall', 'AUC', 'PR-AUC', 'LogLoss']
END_METRICS = ['Best_Epoch', 'Time per Epoch', 'Infer Time', 'DIR']

def get_metrics():
    """Get complete metrics list including top-k metrics"""
    metrics = BASE_METRICS.copy()
    for k in TOP_KS:
        metrics.extend([f'Precision@{k}', f'Recall@{k}', f'MAP@{k}', f'NDCG@{k}'])
    metrics.extend(END_METRICS)
    return metrics

def get_optimizer_args():
    """Get optimizer arguments"""
    return {
        'optimizer': OPTIMIZER,
        'lr': LR,
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY
    }

def get_lr_scheduler_args():
    """Get learning rate scheduler arguments"""
    return {
        'lr_scheduler': LR_SCHEDULER,
        'T_max': T_MAX,
        'T_0': T_0,
        'eta_min': ETA_MIN
    }

def get_device():
    """Get device configuration"""
    return torch.device("cuda:" + str(DEVICES[0])) if torch.cuda.is_available() else torch.device("cpu") 