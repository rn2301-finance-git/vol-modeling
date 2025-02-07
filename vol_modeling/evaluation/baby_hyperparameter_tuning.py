import sys
import os
import json

# Add the project root to Python path - handle both possible locations
if os.path.exists("/workspace/vol_project"):
    sys.path.append("/workspace/vol_project")
elif os.path.exists("/ml_data/vol_project"):
    sys.path.append("/ml_data/vol_project")
else:
    raise RuntimeError("Could not find vol_project project directory in expected locations")

import torch
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import uuid
import pandas as pd

# Import your existing modules
from evaluation.preprocessing import load_parquet_data
from evaluation.run_manager import RunManager, setup_logging
from models.sequence_model import train_sequence_model, evaluate_sequence_model
from evaluation.evaluator import ModelEvaluator
from data_pipeline.features_in_model import selected_features, lasso_features

# -------------------------
# Import your XGBoost helpers
# -------------------------
from models.xgboost_model import train_xgboost_model, evaluate_xgboost_model, predict_xgboost_model
from models.deep_learning_model import train_mlp_model, evaluate_mlp_model, get_predictions

# Add import for Lasso model functions
from models.lasso_model import train_lasso_model, evaluate_lasso_model

# --------------------------------------------------------------------
# 1) Define your experiment plan (PHASES) — now includes XGBOOST
# --------------------------------------------------------------------

PHASES = {
    "SEQUENCE": {
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: Sequence Baseline",
                "hidden_dim": 64,
                "learning_rate": 1e-5,
                "dropout": 0.2,
                "batch_size": 512,
                "gradient_accumulation_steps": 4,  # Effective batch size = 2048
                "eval_train_set_frequency": 5,
                "epochs_fraction": 1.0,  # Full epochs
                "sequence_params": {
                    "short_seq_len": 20,
                    "long_seq_len": 60,
                    "sample_every": 10
                },
                "target_scaling": "none"  # Added as specified
            }
        ]
    },
    "MLP": {
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: MLP Baseline",
                "hidden_dims": [64, 32],
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 2048,
                "gradient_accumulation_steps": 1,
                "eval_train_set_frequency": 5,
                "epochs_fraction": 1.0  # Full epochs
            }
        ]
    },
    "XGBOOST": {
        "PHASE1_BASELINE": [
            {
                "description": "Run 1: XGB Baseline",
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "device": "cpu"  # XGBoost specific device setting
            }
        ]
    }
}


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

def main(
    mode: str = "subset",
    test_n: int = 10,
    use_cuda: bool = True,
    experiment_name: Optional[str] = None,
    model_type: str = "seq_mlp",
    debug: bool = False,
    subsample_fraction: Optional[float] = None,
    asymmetric_loss: bool = False
):
    # Generate experiment name with timestamp if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"experiment_{model_type}_{timestamp}"
    else:
        # Append timestamp to provided experiment name if it doesn't have one
        if not any(c.isdigit() for c in experiment_name):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            experiment_name = f"{experiment_name}_{timestamp}"
    
    # Modify experiment name if using asymmetric loss
    if asymmetric_loss:
        experiment_name = f"{experiment_name}_assym"
    
    logger.info(f"Starting experiment: {experiment_name}")

    try:
        # Load data based on model type with optimized settings
        print("Loading data...")
        # For XGBOOST or MLP, we do not need sequence_params
        sequence_params = {
            "short_seq_len": 20,
            "long_seq_len": 60,
            "sample_every": 10,
            "batch_size": 64
        } if model_type in ["seq_mlp", "sequence"] else None

        datasets, feature_scaler, target_scaler = load_parquet_data(
            mode=mode,
            include_test=False,
            test_n=test_n,
            sequence_mode=(model_type in ["seq_mlp", "sequence"]),
            sequence_params=sequence_params,
            max_workers=4,
            chunk_size=10,
            debug=debug,
            model_type=model_type,
            subsample_fraction=subsample_fraction
        )

        # Some simple logging of dataset sizes
        if model_type in ["seq_mlp", "sequence"]:
            logger.info("Data loaded. Train/Val sizes (sequence loaders):")
            logger.info(f"  - train: {len(datasets['train'].dataset)} sequences")
            logger.info(f"  - validate: {len(datasets['validate'].dataset)} sequences")
        else:
            logger.info("Data loaded. Train/Val sizes (DataFrames):")
            logger.info(f"  - train: {len(datasets['train'])} samples")
            logger.info(f"  - validate: {len(datasets['validate'])} samples")

        # A central evaluator
        evaluator = ModelEvaluator(mode=mode)

        # Initialize phases based on model type - now only using PHASE1
        if model_type == "seq_mlp":
            phases = {"PHASE1_BASELINE": PHASES["SEQUENCE"]["PHASE1_BASELINE"]}
        elif model_type == "mlp":
            phases = {"PHASE1_BASELINE": PHASES["MLP"]["PHASE1_BASELINE"]}
        elif model_type == "xgboost":
            phases = {"PHASE1_BASELINE": PHASES["XGBOOST"]["PHASE1_BASELINE"]}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Process the single phase
        phase_key = "PHASE1_BASELINE"
        logger.info("\n" + "="*50)
        logger.info(f"Running {phase_key}")
        logger.info("="*50)
        
        phase_runs = phases[phase_key]
        for run_cfg in phase_runs:
            # Set loss type based on asymmetric_loss parameter
            if model_type in ["seq_mlp", "mlp"]:
                run_cfg["loss_type"] = "asymmetric" if asymmetric_loss else "mse"
            
            # Set device
            if model_type != "xgboost":  # XGBoost already has device in config
                run_cfg["device"] = "cuda" if use_cuda else "cpu"

            # Run experiment
            run_results = run_experiment(
                datasets=datasets,
                evaluator=evaluator,
                config=run_cfg,
                model_type=model_type,
                final_run=True,  # Always treat as final run
                use_cuda=use_cuda,
                experiment_name=experiment_name,
                debug_mode=debug,
                asymmetric_loss=asymmetric_loss
            )

            logger.info(f"Run complete. Results: {run_results}")

    except Exception as e:
        logger.error("Fatal error in main execution", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def run_experiment(
    datasets,
    evaluator,
    config: Dict[str, Any],
    model_type: str,
    final_run: bool = False,
    use_cuda: bool = False,
    experiment_name: str = None,
    debug_mode: bool = False,
    asymmetric_loss: bool = False
):
    """Run a single experiment with given config."""
    desc = config.get("description", "No desc")
    
    # Set up device
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    # Initialize model and val_metrics
    model = None
    val_metrics = {
        "rmse": float('inf'),
        "rank_corr": 0.0,
        "asymm_mse": float('inf')
    }
    
    try:
        # Create a unique sub-experiment name for this run
        run_name = f"{desc.lower().replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}"
        run_base_path = f"experiments/{model_type}/{experiment_name}/{run_name}"
        
        # Create a new RunManager for this specific run
        if model_type in ("mlp", "seq_mlp", "xgboost", "sequence", "lasso"):
            mode_for_run = (
                "full" if (
                    (model_type in ("seq_mlp", "sequence") and len(datasets["train"].dataset) > 1000) or
                    (model_type not in ("seq_mlp", "sequence") and len(datasets["train"]) > 1000)
                ) else "subset"
            )
            
            run_manager = RunManager(
                model_type=model_type,
                mode=mode_for_run,
                base_path=run_base_path
            )
            logger = run_manager.logger
            logger.info(f"Created new run manager at {run_base_path}")
        else:
            run_manager = None
            logger = None

        if logger:
            logger.info(f"Starting experiment: {desc}")

        # Calculate actual epochs based on fraction and debug mode
        base_epochs = 5 if debug_mode else (
            50 if model_type in ["xgboost", "lasso"] else 100  # Add lasso to fewer epochs group
        )
        epochs = int(base_epochs * config.get("epochs_fraction", 1.0))

        # Set early stopping parameters for neural network models
        early_stopping_params = {
            "patience": 10,
            "min_delta": 1e-4,
            "mode": "min"
        } if model_type in ["mlp", "seq_mlp", "sequence"] else None

        if model_type == "mlp":
            model, target_scaler = train_mlp_model(
                train_df=datasets["train"],
                val_df=datasets["validate"],
                feature_cols=selected_features,
                epochs=epochs,
                batch_size=config.get("batch_size", 512),
                learning_rate=config.get("learning_rate", 3e-4),
                hidden_dims=config.get("hidden_dims", [128, 64]),
                dropout=config.get("dropout", 0.2),
                use_cuda=use_cuda,
                run_manager=run_manager,
                eval_train_set_frequency=config.get("eval_train_set_frequency", 5),
                asymmetric_loss=asymmetric_loss,
                early_stopping_params=early_stopping_params
            )
            
            # Clean up CUDA memory
            if torch.cuda.is_available():
                if hasattr(model, 'cpu'):
                    model.cpu()
                torch.cuda.empty_cache()

            # Save final model state and config
            if run_manager:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=None,  # No need for optimizer state in final save
                    epoch=-1,  # Use -1 to indicate final state
                    val_loss=val_metrics.get("rmse", float('inf')),
                    val_metrics=val_metrics,
                    is_best=True,
                    feature_scaler=datasets.get("feature_scaler"),
                    target_scaler=target_scaler
                )

            # Evaluate final model
            evaluate_mlp_model(
                model=model,
                train_df=datasets["train"],
                val_df=datasets["validate"],
                evaluator=evaluator,
                scaler=target_scaler,
                feature_cols=selected_features,
                notes=config.get("description", ""),
                use_cuda=use_cuda
            )
            # After evaluate_mlp_model, get predictions and evaluate
            val_preds, val_targets = get_predictions(model, datasets["validate"], selected_features, target_scaler, device)
            # Create a DataFrame with true and predicted values
            val_pred_df = pd.DataFrame({
                'Y_log_vol_10min_lag_1m': datasets["validate"]['Y_log_vol_10min_lag_1m'],
                'predicted_vol': val_preds
            })
            val_metrics = evaluator.evaluate_predictions(
                val_pred_df,
                y_true_col='Y_log_vol_10min_lag_1m',
                y_pred_col='predicted_vol'
            )

        elif model_type in ["sequence", "seq_mlp"]:
            model, target_scaler = train_sequence_model(
                train_loader=datasets["train"],
                val_loader=datasets["validate"],
                feature_cols=selected_features,
                epochs=epochs,
                learning_rate=config.get("learning_rate", 3e-4),
                hidden_dims=config.get("hidden_dims", [64, 32]),
                hidden_dim=config.get("hidden_dim", 64),
                dropout=config.get("dropout", 0.2),
                use_cuda=use_cuda,
                sequence_params=config.get("sequence_params", {
                    "short_seq_len": 20,
                    "long_seq_len": 60,
                    "sample_every": 10,
                    "batch_size": 64
                }),
                run_manager=run_manager,
                eval_train_set_frequency=config.get("eval_train_set_frequency", 5),
                early_stopping_params=early_stopping_params,
                asymmetric_loss=asymmetric_loss
            )
            
            # Save checkpoint with both scalers
            if run_manager:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=None,
                    epoch=-1,
                    val_loss=val_metrics.get("rmse", float('inf')),
                    val_metrics=val_metrics,
                    is_best=True,
                    feature_scaler=datasets.get("feature_scaler"),
                    target_scaler=target_scaler
                )

            # Evaluate final model
            val_metrics = evaluator.evaluate_sequence_model(
                model=model,
                data_loader=datasets["validate"],
                device=device
            )

        elif model_type == "xgboost":
            # Train XGBoost
            n_estimators = config.get("n_estimators", 200)
            lr = config.get("learning_rate", 0.1)
            max_depth = config.get("max_depth", 6)
            subsample = config.get("subsample", 0.8)
            colsample_bytree = config.get("colsample_bytree", 0.8)

            # Possibly reduce n_estimators if debug
            if debug_mode:
                n_estimators = 50

            model, target_scaler = train_xgboost_model(
                train_df=datasets["train"],
                val_df=datasets["validate"],
                feature_cols=selected_features,
                target_col="Y_log_vol_10min_lag_1m",
                n_estimators=n_estimators,
                learning_rate=lr,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                use_cuda=use_cuda,
                run_manager=run_manager,
                asymmetric_loss=asymmetric_loss
            )

            # Save checkpoint with both scalers
            if run_manager:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=None,
                    epoch=-1,
                    val_loss=val_metrics.get("rmse", float('inf')),
                    val_metrics=val_metrics,
                    is_best=True,
                    feature_scaler=datasets.get("feature_scaler"),
                    target_scaler=target_scaler
                )

            # Evaluate
            val_pred_df = predict_xgboost_model(
                model=model,
                df=datasets["validate"],
                feature_cols=selected_features,
                target_scaler=target_scaler,
                target_col="Y_log_vol_10min_lag_1m"
            )
            val_metrics = evaluator.evaluate_predictions(
                val_pred_df,
                y_true_col="Y_log_vol_10min_lag_1m",
                y_pred_col="predicted_vol"
            )

        elif model_type == "lasso":
            # Train Lasso model
            model, target_scaler = train_lasso_model(
                train_df=datasets["train"],
                val_df=datasets["validate"],
                feature_cols=lasso_features,  # Use lasso_features
                alpha=config.get("alpha", 1.0),
                max_iter=config.get("max_iter", 1000),
                tol=config.get("tol", 1e-4),
                run_manager=run_manager
            )

            # Save checkpoint with both scalers
            if run_manager:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=None,
                    epoch=-1,
                    val_loss=val_metrics.get("rmse", float('inf')),
                    val_metrics=val_metrics,
                    is_best=True,
                    feature_scaler=datasets.get("feature_scaler"),
                    target_scaler=target_scaler
                )

            # Evaluate Lasso model
            val_metrics = evaluate_lasso_model(
                model=model,
                train_df=datasets["train"],
                val_df=datasets["validate"],
                evaluator=evaluator,
                target_scaler=target_scaler,
                feature_cols=lasso_features,  # Use lasso_features
                notes=config.get("description", "")
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # -------------------------------------------
        # 3) Log results
        # -------------------------------------------
        if logger:
            logger.info(f"Finished run: {desc} | Val RMSE={val_metrics.get('rmse', float('inf')):.4f} | rank_corr={val_metrics.get('rank_corr', 0.0):.4f}")
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
        # Clean up resources
        if model is not None:
            del model
        if run_manager:
            run_manager.cleanup_logging()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Return metrics
    return {
        "description": desc,
        "val_rmse": val_metrics.get("rmse", float('inf')),
        "val_rank_corr": val_metrics.get("rank_corr", 0.0),
        "val_asymm_mse": val_metrics.get("asymm_mse", float('inf'))
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning experiments (MLP, Sequence, XGB, Lasso).')
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
    parser.add_argument('-m', '--model-type', type=str, default="seq_mlp",
                       choices=['seq_mlp', 'mlp', 'xgboost', 'lasso'],
                       help='Type of model to tune (default: seq_mlp)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (only first phase, fewer epochs/trees)')
    parser.add_argument('-f', '--subsample-fraction', type=float,
                       help='Fraction of data to use (0.0 to 1.0). Cannot be used with sequence models.')
    parser.add_argument('-a', '--asymmetric-loss', action='store_true',
                       help='Use asymmetric loss function instead of MSE')
    
    args = parser.parse_args()
    
    # Validate subsample_fraction if provided
    if args.subsample_fraction is not None:
        if args.model_type in ['seq_mlp', 'sequence']:
            parser.error("--subsample-fraction cannot be used with sequence models")
        if not 0.0 < args.subsample_fraction <= 1.0:
            parser.error("--subsample-fraction must be between 0.0 and 1.0")
    
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
            asymmetric_loss=args.asymmetric_loss
        )
    except Exception as e:
        logger.error("Fatal error in main execution", exc_info=True)
        raise
