"""
Hyperparameter tuning framework for various models including transformers, XGBoost, and MLPs.

This module provides a flexible framework for:
- Running hyperparameter tuning experiments
- Handling multiple model types
- Managing experiment logging and checkpointing
- Evaluating model performance

Example usage:
    python hyperparameter_tuning.py --model-type transformer --mode subset --test-n 10
"""

import sys
import os
import json
import pathlib

# Add the project root to Python path - handle both possible locations
if os.path.exists("/workspace/BAM"):
    sys.path.append("/workspace/BAM")
elif os.path.exists("/ml_data/BAM"):
    sys.path.append("/ml_data/BAM")
else:
    raise RuntimeError("Could not find BAM project directory in expected locations")

from sklearn.discriminant_analysis import StandardScaler
import torch
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import argparse
import uuid
import pandas as pd

# Import your existing modules
from evaluation.preprocessing import load_parquet_data
from evaluation.run_manager import RunManager, setup_logging
from models.sequence_model import train_sequence_model
from models.three_headed_transformer import train_three_headed_model, evaluate_three_headed_model
from evaluation.evaluator import ModelEvaluator
from data_pipeline.features_in_model import selected_features, lasso_features

# -------------------------
# Import your XGBoost helpers
# -------------------------
from models.xgboost_model import train_xgboost_model, evaluate_xgboost_model, predict_xgboost_model
from models.deep_learning_model import train_mlp_model, evaluate_mlp_model, get_predictions

# Add import for Lasso model functions
from models.lasso_model import train_lasso_model, evaluate_lasso_model

from evaluation.eval_phases import TEMP_PHASES
# --------------------------------------------------------------------
# 1) Define your experiment plan (PHASES) — now includes XGBOOST
# --------------------------------------------------------------------
PHASES = TEMP_PHASES #just for testing

def parse_phase_key(phase_key: str) -> tuple:
    """
    Convert a phase name like "PHASE2_LR_TEST" into (2, "LR_TEST"),
    so we can sort phases numerically first, then alphabetically.
    """
    prefix = "PHASE"
    if not phase_key.startswith(prefix):
        return (9999, phase_key)
    remainder = phase_key[len(prefix):]
    parts = remainder.split("_", 1)
    try:
        phase_num = int(parts[0])
    except ValueError:
        phase_num = 9999
    label = parts[1] if len(parts) > 1 else ""
    return (phase_num, label)

def validate_model_type(model_type, phases):
    """Validate and normalize model type."""
    # Create case-insensitive mapping of model types
    model_map = {k.lower(): k for k in phases.keys()}
    
    # Look up using lowercase version
    model_type_lower = model_type.lower()
    if model_type_lower not in model_map:
        raise ValueError(f"Model type {model_type} not found in phases. Available types: {list(phases.keys())}")
    
    # Return the correct case version from phases
    return model_map[model_type_lower]

def update_final_phase_config(phases, phase_results):
    """Update PHASE5_FINAL with best parameters from previous phases."""
    best_config = None
    best_performance = float('inf')
    
    # Find best configuration based on validation RMSE
    for result in phase_results:
        if result['val_rmse'] < best_performance:
            best_performance = result['val_rmse']
            best_config = result.get('config', {})
    
    # Update final phase with best parameters if it exists
    if 'PHASE5_FINAL' in phases:
        final_phase = phases['PHASE5_FINAL'][0]  # Assuming single config in final phase
        for param in final_phase:
            if final_phase[param] is None and param in best_config:
                final_phase[param] = best_config[param]

def get_project_root() -> pathlib.Path:
    """Get the project root directory."""
    current_file = pathlib.Path(__file__).resolve()
    return current_file.parent.parent

# Replace the current path checking code with:
project_root = get_project_root()
sys.path.append(str(project_root))

def main(
    mode: str = "subset",
    test_n: int = 10,
    use_cuda: bool = True,
    experiment_name: Optional[str] = None,
    model_type: str = "transformer",
    debug: bool = False,
    subsample_fraction: Optional[float] = None,
    asymmetric_loss: bool = False,
    target: str = "vol",
    use_lagged_targets: bool = False
):
    # Generate structured experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"hyperparameter_tuning_{model_type}_{target}_{timestamp}"
    
    # Add debug/asymmetric indicators to experiment name
    if debug:
        experiment_name += "_debug"
    if asymmetric_loss:
        experiment_name += "_asymmetric"
    if not use_lagged_targets:
        experiment_name += "_nolags"
    
    logger.info(f"Starting experiment: {experiment_name}")
    
    try:
        # Load data based on model type with optimized settings
        print("Loading data...")
        sequence_params = None
        
        # Set sequence parameters for sequence-based models
        if model_type in ["seq_mlp", "transformer"]:
            sequence_params = {
                "sequence_length": 30,
                "sample_every": 10,
                "batch_size": 64 if model_type == "transformer" else 8192
            }
        if model_type in ["seq_mlp"]:
            if subsample_fraction is not None:
                logger.warning("subsample_fraction ignored for sequence models")
                subsample_fraction = None
        
        # Load and preprocess data - pass debug flag to ensure TEST_SPLITS usage
        datasets, feature_scaler, target_scalers = load_parquet_data(
            mode=mode,
            test_n=test_n,
            model_type=model_type,
            sequence_mode=(model_type in ["seq_mlp", "transformer"]),
            sequence_params=sequence_params,
            subsample_fraction=subsample_fraction,
            debug=debug,  # This ensures TEST_SPLITS is used when debug=True
            use_lagged_targets=use_lagged_targets  # Add this parameter
        )

        # Initialize evaluator
        evaluator = ModelEvaluator(mode=mode)
        
        # Validate model type and get phases
        model_type = validate_model_type(model_type, PHASES)
        model_phases = PHASES[model_type]
        sorted_phases = sorted(model_phases.keys(), key=parse_phase_key)
        
        if debug:
            sorted_phases = sorted_phases[:1]
            for phase in sorted_phases:
                for config in model_phases[phase]:
                    if model_type == "XGBOOST":
                        config["n_estimators"] = min(config.get("n_estimators", 100), 10)
                    elif "epochs" in config:
                        config["epochs"] = min(config["epochs"], 3)
        
        # Track all phase results
        all_phase_results = []
        
        # Run experiments for each phase
        for phase in sorted_phases:
            logger.info(f"\nStarting {phase}")
            phase_results = []
            
            for config in model_phases[phase]:
                # Log phase configuration
                logger.info(f"\nPhase configuration:")
                logger.info(f"Phase: {phase}")
                for key, value in config.items():
                    if key != 'description':
                        logger.info(f"{key}: {value}")
                
                # Add asymmetric loss parameters if enabled
                if asymmetric_loss:
                    config["asymmetric_loss"] = True
                    config["asymmetric_alpha"] = 1.5
                
                result = run_experiment(
                    datasets=datasets,
                    evaluator=evaluator,
                    config=config,
                    model_type=model_type,
                    use_cuda=use_cuda,
                    experiment_name=experiment_name,
                    debug_mode=debug,
                    asymmetric_loss=asymmetric_loss,
                    target=target,
                    feature_scaler=feature_scaler,
                    use_lagged_targets=use_lagged_targets
                )
                
                # Store config with result for parameter inheritance
                result['config'] = deepcopy(config)
                phase_results.append(result)
                all_phase_results.append(result)
            
            # Update final phase configuration if this isn't the final phase
            if phase != "PHASE5_FINAL":
                update_final_phase_config(model_phases, all_phase_results)
            
            # Log phase results
            logger.info(f"\nPhase {phase} Results:")
            for result in phase_results:
                logger.info(f"Config: {result['description']}")
                logger.info(f"Val RMSE: {result['val_rmse']:.4f}")
                logger.info(f"Rank Correlation: {result['val_rank_corr']:.4f}")
                if asymmetric_loss:
                    logger.info(f"Asymmetric MSE: {result['val_asymm_mse']:.4f}")
                logger.info("-" * 40)
        
        logger.info("\nExperiment completed successfully!")
        
    except Exception as e:
        logger.error("Fatal error in main execution", exc_info=True)
        raise

def run_experiment(
    datasets,
    evaluator: ModelEvaluator,
    config: Dict[str, Any],
    model_type: str,
    final_run: bool = False,
    use_cuda: bool = True,
    experiment_name: Optional[str] = None,
    debug_mode: bool = False,
    asymmetric_loss: bool = False,
    asymmetric_alpha: float = 1.5,
    target: str = "vol",
    feature_scaler: Any = None,
    use_lagged_targets: bool = False
) -> Dict[str, float]:
    """Run a single experiment with given configuration."""
    logger = logging.getLogger('model_training')
    model = None
    run_manager = None
    
    try:
        # Initialize RunManager with all necessary parameters
        base_path = os.path.join("experiments", model_type.lower())  # Use lowercase for paths
        os.makedirs(base_path, exist_ok=True)
        
        run_manager = RunManager(
            model_type=model_type,
            mode="test" if debug_mode else "prod",
            experiment_name=experiment_name,
            base_path=base_path
        )

        # Handle both string and dict configs
        if isinstance(config, str):
            desc = config
            model_params = {}  # Empty dict for string configs
        else:
            desc = config.get("description", "")
            # Create a deep copy of config to preserve all parameters
            model_params = deepcopy(config)
            # Remove description if it exists, but keep all other parameters
            model_params.pop("description", None)

        # Save configuration information
        run_manager.save_config(
            args={
                "model_type": model_type,
                "debug_mode": debug_mode,
                "use_cuda": use_cuda,
                "experiment_name": experiment_name,
                "asymmetric_loss": asymmetric_loss,
                "asymmetric_alpha": asymmetric_alpha,
                "target": target
            },
            dataset_info={
                "train_size": len(datasets["train"]),
                "val_size": len(datasets["validate"]),
                "features": selected_features,
            },
            model_params=model_params
        )
        

        
        # Log experiment configuration
        logger.info("\n" + "="*80)
        logger.info(f"Starting experiment: {desc}")
        logger.info(f"Model type: {model_type}")
        logger.info("="*80 + "\n")
        
        if model_type.lower() == "transformer":
            # Import transformer-specific functions
            from models.three_headed_transformer import train_three_headed_model, evaluate_three_headed_model
            
            # Get feature dimension from the data
            if isinstance(datasets["train"], torch.utils.data.DataLoader):
                sample_batch = next(iter(datasets["train"]))
                input_dim = sample_batch[0].shape[-1]
            else:
                input_dim = len(selected_features)
            
            logger.info(f"Using input dimension: {input_dim}")
            
            # Extract sequence parameters from the datasets if available
            sequence_params = getattr(datasets.get("train"), "sequence_params", {
                "sequence_length": 30,
                "sample_every": 10
            })
            
            # Prepare training configuration
            training_config = {
                # Required parameters
                "input_dim": input_dim,
                "train_loader": datasets["train"],
                "val_loader": datasets["validate"],
                "run_manager": run_manager,
                "use_cuda": use_cuda,
                "debug": debug_mode,
                "feature_scaler": feature_scaler,
                # Optional parameters with defaults
                "hidden_dim": config.get("hidden_dim", 256),
                "nhead": config.get("nhead", 8),
                "num_layers": config.get("num_layers", 3),
                "dropout": config.get("dropout", 0.1),
                "learning_rate": config.get("learning_rate", 1e-4),
                "epochs": config.get("epochs", 100),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
                "sequence_params": sequence_params,
                "eval_train_set_frequency": config.get("eval_train_set_frequency", 5),
                "asymmetric_loss": asymmetric_loss,
                "early_stopping_params": config.get("early_stopping_params", {
                    "patience": 10,
                    "min_delta": 1e-4,
                    "mode": "min"
                }),
                "gamma": config.get("gamma", 0.1),
                "weight_decay": config.get("weight_decay", 0.01),
                "use_parameter_groups": config.get("use_parameter_groups", False)
            }
            
            # Remove any None values and use_lagged_targets (since it's not used directly by train_three_headed_model)
            training_config = {k: v for k, v in training_config.items() if v is not None and k != 'use_lagged_targets'}
            
            model, target_scalers = train_three_headed_model(**training_config)
            
            # Evaluate transformer model with model parameter
            val_metrics = evaluate_three_headed_model(
                model=model,
                train_loader=datasets["train"],
                val_loader=datasets["validate"],
                evaluator=evaluator,
                use_cuda=use_cuda,
                notes=desc,
            )
            
            # Save best model checkpoint with both scalers
            if run_manager:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=None,
                    epoch=-1,
                    val_loss=val_metrics.get("rmse", float('inf')),
                    val_metrics=val_metrics,
                    is_best=True,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scalers
                )
                
        elif model_type.lower() == "xgboost":
            # Create a copy of config without the description and feature_cols fields
            xgb_params = {k: v for k, v in config.items() if k not in ['description', 'feature_cols']}
            
            # Modify target column selection based on use_lagged_targets
            target_col = ("Y_log_ret_60min_lag_1m" if use_lagged_targets else "Y_log_ret_60min") if target == "ret" else ("Y_log_vol_10min_lag_1m" if use_lagged_targets else "Y_log_vol_10min")
            
            model, target_scaler = train_xgboost_model(
                train_df=datasets["train"],
                val_df=datasets["validate"],
                feature_cols=selected_features,
                run_manager=run_manager,
                target=target,
                target_col=target_col,  # Use correct target column
                **xgb_params  # Use filtered parameters
            )
            
            val_metrics = evaluate_xgboost_model(
                model=model,
                train_df=datasets["train"],
                val_df=datasets["validate"],
                evaluator=evaluator,
                target_scaler=target_scaler,
                feature_cols=selected_features,
                target_col=target_col,  # Use same target column
                notes=desc
            )

            
            val_metrics = evaluate_xgboost_model(
                model=model,
                train_df=datasets["train"],
                val_df=datasets["validate"],
                evaluator=evaluator,
                target_scaler=target_scaler,
                feature_cols=selected_features,
                target_col=target_col,  # Use same target column
                notes=desc
            )
            
        elif model_type.lower() == "mlp":
            model, target_scaler = train_mlp_model(
                train_df=datasets["train"],
                val_df=datasets["validate"],
                feature_cols=selected_features,
                run_manager=run_manager,
                use_cuda=use_cuda,
                target=target,
                **config
            )
            
            val_metrics = evaluate_mlp_model(
                model=model,
                train_df=datasets["train"],
                val_df=datasets["validate"],
                evaluator=evaluator,
                target_scaler=target_scaler,
                feature_cols=selected_features,
                target=target,
                notes=desc
            )
            
            if run_manager:
                final_metrics = {
                    'train_rmse': val_metrics.get('train_rmse', float('inf')),
                    'train_rank_corr': val_metrics.get('train_rank_corr', 0.0),
                    'val_rmse': val_metrics.get('rmse', float('inf')),
                    'val_rank_corr': val_metrics.get('rank_corr', 0.0),
                    'val_asymm_mse': val_metrics.get('asymm_mse', float('inf'))
                }
                run_manager.save_final_metrics(final_metrics)
            
        elif model_type.lower() == "seq_mlp":
            model, target_scaler = train_sequence_model(
                train_loader=datasets["train"],
                val_loader=datasets["validate"],
                run_manager=run_manager,
                use_cuda=use_cuda,
                target=target,
                **config
            )
            
            val_metrics = evaluate_sequence_model(
                model=model,
                train_loader=datasets["train"],
                val_loader=datasets["validate"],
                evaluator=evaluator,
                target_scaler=target_scaler,
                notes=desc
            )
            
            if run_manager:
                final_metrics = {
                    'train_rmse': val_metrics.get('train_rmse', float('inf')),
                    'train_rank_corr': val_metrics.get('train_rank_corr', 0.0),
                    'val_rmse': val_metrics.get('rmse', float('inf')),
                    'val_rank_corr': val_metrics.get('rank_corr', 0.0),
                    'val_asymm_mse': val_metrics.get('asymm_mse', float('inf'))
                }
                run_manager.save_final_metrics(final_metrics)
            
        elif model_type.lower() == "lasso":
            model, target_scaler = train_lasso_model(
                train_df=datasets["train"],
                val_df=datasets["validate"],
                feature_cols=lasso_features,  # Using lasso_features for Lasso model
                run_manager=run_manager,
                target=target,
                **config
            )
            
            val_metrics = evaluate_lasso_model(
                model=model,
                train_df=datasets["train"],
                val_df=datasets["validate"],
                evaluator=evaluator,
                target_scaler=target_scaler,
                feature_cols=lasso_features,
                target=target,
                notes=desc
            )
            
            if run_manager:
                final_metrics = {
                    'train_rmse': val_metrics.get('train_rmse', float('inf')),
                    'train_rank_corr': val_metrics.get('train_rank_corr', 0.0),
                    'val_rmse': val_metrics.get('rmse', float('inf')),
                    'val_rank_corr': val_metrics.get('rank_corr', 0.0),
                    'val_asymm_mse': val_metrics.get('asymm_mse', float('inf'))
                }
                run_manager.save_final_metrics(final_metrics)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Log results
        if logger:
            logger.info(f"\nFinished run: {desc}")
            logger.info(f"Val RMSE={val_metrics.get('rmse', float('inf')):.4f}")
            logger.info(f"Rank correlation={val_metrics.get('rank_corr', 0.0):.4f}")
        
        return {
            "description": desc,
            "val_rmse": val_metrics.get("rmse", float('inf')),
            "val_rank_corr": val_metrics.get("rank_corr", 0.0),
            "val_asymm_mse": val_metrics.get("asymm_mse", float('inf'))
        }

    except Exception as e:
        if logger:
            logger.error(f"Error in experiment: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up resources safely
        if model is not None:
            del model
        if run_manager:
            try:
                if hasattr(run_manager, 'cleanup_logging'):
                    run_manager.cleanup_logging()
                if hasattr(run_manager, 'cleanup'):
                    run_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error during RunManager cleanup: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_model_params(model_type: str, use_cuda: bool = False, returns_fit: bool = False, use_lagged_targets: bool = False) -> Dict:
    # ... existing code ...

    params = {
        # ... other model params ...
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
            'gamma': 0.1,  # Add default gamma value
            'target_col': ret_target if returns_fit else vol_target,
            'sequence_params': {
                'sequence_length': 30,
                'sample_every': 10,
                'batch_size': 64
            }
        },
        # ... other models ...
    }
    return params.get(model_type, {})

if __name__ == "__main__":
    # Move logger setup after argument parsing
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning experiments (MLP, Sequence, XGB, Lasso, Transformer).')
    parser.add_argument('-d', '--mode', type=str, default="subset",
                       choices=['full', 'subset'],
                       help='Whether to use full or subset of data')
    parser.add_argument('-t', '--test-n', type=int, default=10,
                       choices=[10, 100],
                       help='Size of subset to use (10 or 100, default: 10)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    parser.add_argument('-e', '--experiment-name', type=str,
                       help='Name for the experiment group (default: timestamp-based)')
    parser.add_argument('-m', '--model-type', type=str, default="transformer",
                       choices=['seq_mlp', 'mlp', 'xgboost', 'lasso', 'transformer'],
                       help='Type of model to tune (default: transformer)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (only first phase, fewer epochs/trees)')
    parser.add_argument('-f', '--subsample-fraction', type=float,
                       help='Fraction of data to use (0.0 to 1.0). Cannot be used with sequence models.')
    parser.add_argument('-a', '--asymmetric-loss', action='store_true',
                       help='Use asymmetric loss function instead of MSE')
    parser.add_argument('-g', '--target', type=str, default="vol",
                       choices=['vol', 'ret'],
                       help='Target to predict (volatility or returns, default: vol)')
    parser.add_argument('-l', '--use-lagged-targets', action='store_true',
                       help='Use lagged targets instead of standard targets')
    
    args = parser.parse_args()
    
    # Validate subsample_fraction if provided
    if args.subsample_fraction is not None:
        if args.model_type in ['seq_mlp', 'sequence']:  # Remove 'transformer' from this list
            parser.error("--subsample-fraction cannot be used with sequence models")
        if not 0.0 < args.subsample_fraction <= 1.0:
            parser.error("--subsample-fraction must be between 0.0 and 1.0")
    
    # Only setup logging if we're actually running the experiment
    # (not just showing help)
    if len(sys.argv) > 1 and sys.argv[1] != '-h' and sys.argv[1] != '--help':
        # Setup logging first
        logger = setup_logging(test_mode=(args.mode == "subset"))
        
        try:
            main(
                mode=args.mode,
                test_n=args.test_n,
                use_cuda=not args.no_cuda,
                experiment_name=args.experiment_name,
                model_type=args.model_type,
                debug=args.debug,
                subsample_fraction=args.subsample_fraction,
                asymmetric_loss=args.asymmetric_loss,
                target=args.target,
                use_lagged_targets=args.use_lagged_targets
            )
        except Exception as e:
            logger.error("Fatal error in main execution", exc_info=True)
            raise
