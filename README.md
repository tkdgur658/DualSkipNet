# Micro Video Recommendation System

This repository contains a micro video recommendation system implementation with configurable training and inference modes.

## Project Structure

```
├── main.py              # Main execution script
├── config.py            # Configuration parameters
├── utils.py             # Utility functions
├── models/              # Model implementations
├── output/              # Output directory for results
├── graphs/              # Graph data directory
└── README.md           # This file
```

## Quick Start

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- tqdm

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch numpy pandas tqdm
```

### Configuration

Edit `config.py` to modify experiment settings:

- **Training Mode**: Set `TRAIN_MODE = 'train'` for training or `'inference'` for inference
- **Dataset**: Modify `DATASET_NAME_LIST` to use different datasets
- **Model**: Change `MODEL_NAMES` to use different models
- **Training Parameters**: Adjust `EPOCHS`, `BATCH_SIZE`, `LR`, etc.

### Running Experiments

#### Training Mode
```bash
python main.py
```

#### Inference Mode
1. Set `TRAIN_MODE = 'inference'` in `config.py`
2. Update `PAST_RESULT_CSV` path to point to your previous results
3. Run:
```bash
python main.py
```

## Output Structure

Results are saved in the `output/` directory with the following structure:
```
output/
└── output_YYYYMMDD_HHMMSS/
    ├── model_dataset_split_ratio_iter/
    │   └── timestamp_model_dataset_split_ratio_iter.pt
    └── Micro_Video_Recommendation_YYYYMMDD_HHMMSS.csv
```

## Configuration Options

### Training Parameters
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LR`: Learning rate
- `EARLY_STOP`: Early stopping patience

### Model Parameters
- `EMBED_DIM`: Embedding dimension
- `MODEL_NAMES`: List of models to train

### Evaluation Parameters
- `TOP_KS`: List of k values for top-k evaluation
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: Data split ratios

## Results

The system outputs comprehensive evaluation metrics including:
- Precision@k, Recall@k, MAP@k, NDCG@k for various k values
- AUC, PR-AUC, LogLoss
- Training time and inference time
- Best epoch information

Results are automatically saved to CSV files in the output directory. 