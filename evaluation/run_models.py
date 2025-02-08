import sys
import os
if os.path.exists("/workspace/vol_modeling"):
    sys.path.append("/workspace/vol_modeling")
elif os.path.exists("/ml_data/vol_project"):
    sys.path.append("/ml_data/vol_project")
else:
    raise RuntimeError("Could not find vol_project project directory in expected locations")
import pandas as pd
import s3fs
from datetime import datetime
from typing import Dict, Optional
import logging
import time
from models.deep_learning_model import train_mlp_model, evaluate_mlp_model
from models.naive_vol_model import train_and_validate_naive
from evaluation.evaluator import ModelEvaluator
from data_pipeline.features_in_model import lasso_features, selected_features
from evaluation.preprocessing import load_parquet_data, preprocess_data
from models.sequence_model import train_sequence_model
import torch
from evaluation.run_manager import RunManager, setup_logging
from models.xgboost_model import train_xgboost_model, evaluate_xgboost_model
from evaluation.hyperparameter_tuning import run_experiment
from models.lasso_model import train_lasso_model, evaluate_lasso_model

BUCKET_NAME = os.environ.get('BUCKET_NAME')
if not BUCKET_NAME:
    raise ValueError("Environment variable BUCKET_NAME must be set")

def upload_debug_dfs(datasets: Dict[str, pd.DataFrame], timestamp: str):
    """Upload dataframes to S3 for debugging."""
    logger = logging.getLogger('model_training')
    logger.info("\nUploading dataframes for debugging...")
    
    fs = s3fs.S3FileSystem(anon=False)
    base_path = f"s3://{BUCKET_NAME}/debug_upload_df"
    
    for split, df in datasets.items():
        file_path = f"{base_path}/{split}_df_{timestamp}.parquet"
        with fs.open(file_path, 'wb') as f:
            df.to_parquet(f)
        logger.info(f"Uploaded {split} df to {file_path}")

def get_model_params(model_type: str, use_cuda: bool = False, returns_fit: bool = False, use_lagged_targets: bool = False) -> Dict:
    """
    Get default parameters for each model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model to get parameters for
    use_cuda : bool
        Whether to use CUDA for GPU acceleration
    returns_fit : bool
        Whether to fit on returns instead of volatility (XGBoost only)
    use_lagged_targets : bool
        Whether to use lagged targets (default: False)
    """
    # Determine target column based on parameters
    vol_target = "Y_log_vol_10min_lag_1m" if use_lagged_targets else "Y_log_vol_10min"
    ret_target = "Y_log_ret_60min_lag_1m" if use_lagged_targets else "Y_log_ret_60min"
    
    logger = logging.getLogger('model_training')
    logger.info(f"\nConfiguring model with {'lagged' if use_lagged_targets else 'non-lagged'} targets:")
    logger.info(f"Volatility target: {vol_target}")
    logger.info(f"Returns target: {ret_target}")
    
    params = {
        'naive': {},
        'mlp': {
            'feature_cols': selected_features,
            'epochs': 100,
            'batch_size': 8192,
            'learning_rate': 1e-3,
            'hidden_dims': [64, 32],
            'dropout': 0.1,
            'use_cuda': use_cuda,
            'target_col': ret_target if returns_fit else vol_target
        },
        'seq_mlp': {
            'feature_cols': selected_features,
            'epochs': 100,
            'learning_rate': 1e-5,
            'hidden_dim': 64,
            'dropout': 0.1,
            'use_cuda': use_cuda,
            'target_col': ret_target if returns_fit else vol_target,
            'sequence_params': {
                'short_seq_len': 20,
                'long_seq_len': 60,
                'sample_every': 10,
                'batch_size': 8192
            }
        },
        'transformer': {
            'feature_cols': selected_features,
            'epochs': 100,
            'learning_rate': 1e-4,
            'hidden_dim': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.4,
            'use_cuda': use_cuda,
            'weight_decay': 0.01,
            'use_parameter_groups': True,
            'target_col': ret_target if returns_fit else vol_target,
            'sequence_params': {
                'sequence_length': 30,
                'sample_every': 10,
                'batch_size': 64
            }
        },
        'xgboost': {
            'feature_cols': selected_features,
            'target_col': ret_target if returns_fit else vol_target,
            'params': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'random_state': 42
            }
        },
    }
    return params.get(model_type, {})

def get_dataset_mode_str(mode: str, test_n: int = None) -> str:
    """Get a descriptive string for the dataset mode."""
    if mode == "full":
        return "FULL"
    elif mode == "subset":
        return f"SUBSET_TOP_{test_n}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main(
    model_type: str = "xgboost",
    mode: str = "subset",
    test_n: int = 10,
    include_test: bool = False,
    upload_df: bool = False,
    debug: bool = True,
    use_cuda: bool = True,
    experiment_name: Optional[str] = None,
    subsample_fraction: Optional[float] = None,
    asymmetric_loss: bool = False,
    asymmetric_alpha: float = 1.5,
    returns_fit: bool = False,
    use_lagged_targets: bool = False
):
    """Main training function."""
    try:
        # Generate experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            experiment_name = f"single_run_{model_type}_{timestamp}"
        
        # Append _retfit for returns fitting mode
        if returns_fit:
            if model_type != 'xgboost':
                raise ValueError("Returns fitting mode is only supported for XGBoost")
            experiment_name = f"{experiment_name}_retfit"
        
        # Append _nolags if not using lagged targets
        if not use_lagged_targets:
            experiment_name = f"{experiment_name}_nolags"
        
        # Setup logging first to capture all output
        logger = setup_logging(test_mode=(mode == "subset"))
        logger.info("\nStarting data loading process...")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Mode: {mode}")
        
        # Get model parameters with use_lagged_targets
        model_params = get_model_params(model_type, use_cuda, returns_fit, use_lagged_targets)
        
        # Extract sequence parameters if needed
        sequence_params = None
        if model_type in ["seq_mlp", "sequence", "transformer"]:
            sequence_params = model_params['sequence_params']
            logger.info(f"Using sequence parameters: {sequence_params}")

        # Add more detailed logging before loading data
        logger.info("Attempting to load parquet data with parameters:")
        logger.info(f"Mode: {mode}")
        logger.info(f"Include test: {include_test}")
        logger.info(f"Test N: {test_n}")
        logger.info(f"Sequence mode: {model_type in ['seq_mlp', 'sequence', 'transformer']}")
        logger.info(f"Subsample fraction: {subsample_fraction}")

        datasets, feature_scaler, target_scalers = load_parquet_data(
            mode=mode,
            include_test=include_test,
            test_n=test_n,
            sequence_mode=(model_type in ["seq_mlp", "sequence", "transformer"]),
            sequence_params=sequence_params,
            debug=debug,
            model_type=model_type,
            subsample_fraction=subsample_fraction,
            use_lagged_targets=use_lagged_targets
        )
        
        # Upload debug DataFrames if requested
        if upload_df and not model_type in ["seq_mlp", "sequence", "transformer"]:
            upload_debug_dfs(datasets, datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        
        # Create config for run_experiment
        config = {
            "description": f"Single run of {model_type}",
            **model_params
        }
        
        # Add sequence-specific parameters if needed
        #if model_type in ["seq_mlp", "sequence", "transformer"]:
        #    config["sequence_params"] = sequence_params
        
        # Initialize evaluator
        evaluator = ModelEvaluator(mode=mode)
        
        # Modify experiment name if using asymmetric loss
        if asymmetric_loss:
            if model_type in ['seq_mlp', 'mlp']:
                experiment_name = f"{experiment_name}_assym"
            else:
                raise ValueError(f"Asymmetric loss is not supported for {model_type}")
        
        # Run the experiment using the shared run_experiment function
        results = run_experiment(
            datasets=datasets,
            evaluator=evaluator,
            config=config,
            model_type=model_type,
            final_run=True,
            use_cuda=use_cuda,
            experiment_name=experiment_name,
            debug_mode=debug,
            asymmetric_loss=asymmetric_loss,
            asymmetric_alpha=asymmetric_alpha,
            feature_scaler=feature_scaler,
        )
        
        logger.info("\nModel training and evaluation completed successfully!")
        logger.info(f"Final metrics: RMSE={results['val_rmse']:.4f}, Rank Correlation={results['val_rank_corr']:.4f}")
        
    except Exception as e:
        if logger:
            logger.error(f"\nError during model training: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        logger.info("\n" + "="*80)
        logger.info(f"Run completed at {datetime.now()}")
        logger.info("="*80 + "\n")

if __name__ == "__main__":
    import argparse
    print("Running run_models.py")
    
    parser = argparse.ArgumentParser(description='Train and evaluate volatility models')
    parser.add_argument('-m', '--model', type=str, required=True,
                       choices=['naive', 'xgboost', 'mlp', 'seq_mlp', 'lasso', 'transformer'],
                       help='Model type to train')
    parser.add_argument('-d', '--mode', type=str, default="subset",
                       choices=['full', 'subset'],
                       help='Whether to use full or subset of data')
    parser.add_argument('-t', '--test-n', type=int, default=10,
                       choices=[10, 100],
                       help='Size of subset to use (10 or 100, default: 10)')
    parser.add_argument('-f', '--subsample-fraction', type=float,
                       help='Fraction of data to use (0.0 to 1.0). Cannot be used with sequence models.')
    parser.add_argument('--final', action='store_true',
                       help='Include test set evaluation')
    parser.add_argument('-u', '--upload_df', action='store_true',
                       help='Upload dataframes to S3 for debugging (only works with subset mode)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('-e', '--experiment-name', type=str,
                       help='Name for the experiment (default: timestamp-based)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('-a', '--asymmetric-loss', action='store_true',
                       help='Use asymmetric loss function instead of MSE')
    parser.add_argument('--asymmetric-alpha', type=float, default=1.5,
                       help='Factor to penalize under-predictions in asymmetric loss (default: 1.5)')
    parser.add_argument('-r', '--returns-fit', action='store_true',
                       help='Fit on returns instead of volatility (XGBoost only)')
    parser.add_argument('-l', '--use-lagged-targets', action='store_true',
                       help='Use lagged targets instead of standard targets')
    
    args = parser.parse_args()
    
    # Validate returns-fit is only used with XGBoost
    if args.returns_fit and args.model != 'xgboost':
        parser.error("--returns-fit (-r) can only be used with XGBoost model")
    
    # Validate upload_df and subset mode combination early
    if args.upload_df and args.mode != "subset":
        parser.error("--upload_df (-u) can only be used with subset mode")
    
    # Validate subsample_fraction if provided
    if args.subsample_fraction is not None:
        if args.model in ['seq_mlp', 'sequence']:
            parser.error("--subsample-fraction cannot be used with sequence models")
        if not 0.0 < args.subsample_fraction <= 1.0:
            parser.error("--subsample-fraction must be between 0.0 and 1.0")
    
    try:
        main(
            model_type=args.model,
            mode=args.mode,
            test_n=args.test_n,
            include_test=args.final,
            upload_df=args.upload_df,
            use_cuda=not args.no_cuda,
            experiment_name=args.experiment_name,
            debug=args.debug,
            subsample_fraction=args.subsample_fraction,
            asymmetric_loss=args.asymmetric_loss,
            asymmetric_alpha=args.asymmetric_alpha,
            returns_fit=args.returns_fit,
            use_lagged_targets=args.use_lagged_targets
        )
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
        raise