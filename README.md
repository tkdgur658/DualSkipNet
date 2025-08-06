# Exploiting Fine-Grained Skip Behaviors for Micro-video Recommendation (AAAI '25)

This repository contains the official code of "Exploiting Fine-Grained Skip Behaviors for Micro-video Recommendation (AAAI '25)" .

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

- Python 3.11
- requirements.txt

### Running Experiments

#### Training Mode
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
