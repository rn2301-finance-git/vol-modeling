# BAM Volatility Prediction Project

A research project to develop robust models for predicting 10-minute forward volatility, with emphasis on production-ready implementation and infrastructure scalability.

## Project Structure
```
.
├── evaluation/           # Model evaluation and runtime management
├── models/              # Model implementations (XGBoost, LSTM, Transformer)
├── data_pipeline/       # Data preprocessing and feature engineering  
├── inference/           # Production inference pipeline
└── scripts/            # Training and hyperparameter tuning
```

## Quick Start
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Train models
```bash
python evaluation/run_models.py -m transformer --use-lagged-targets
python evaluation/hyperparameter_tuning.py -m xgboost --subsample-fraction 0.2
```

3. Run inference
```bash
python inference/run_inference.py -m transformer -e experiment_name -p run_prefix
```

## Models
- Three-headed Transformer: Joint prediction of volatility, returns and confidence
- XGBoost: Baseline model with proven performance
- Sequence MLP: For capturing temporal patterns
- Lasso: Linear baseline with feature selection

## Key Features
- Asymmetric loss functions prioritizing over-prediction
- Auto-scaling infrastructure with AWS spot instances
- Robust data pipeline with proper forward-fill handling
- Production-grade inference with 5-minute sampling

## Data Specs
- 1-minute BBO data from NASDAQ ITCH
- Top 1000 symbols by liquidity
- Training: Jan-Jun 2024
- Validation: Jul-Aug 2024
- Test: Sep 2024-Jan 2025
