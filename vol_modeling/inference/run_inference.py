#!/usr/bin/env python
import sys
import os

# Add parent directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import argparse
import logging
import time
from datetime import datetime
from typing import Optional, Any, Dict, List
import re
import json

# Third-party imports
import s3fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import torch
import xgboost 
from xgboost import XGBRegressor
import torch.nn as nn
from torch.utils.data import DataLoader

# Import your own modules after modifying sys.path
from data_pipeline.features_in_model import selected_features
from evaluation.preprocessing import handle_missing_values
from models.xgboost_model import predict_xgboost_model
from models.deep_learning_model import VolatilityDataset
from models.lasso_model import predict_lasso_model
from evaluation.run_manager import RunManager, setup_logging
from models.sequence_model import VolForecastNet
from models.three_headed_transformer import predict_three_headed_model
from data_pipeline.sequence_dataset import ThreeHeadedTransformerDataset
from inference.keep_cols import KEEP_COLUMNS

###############################################################################
# Utility to set up logging
###############################################################################
def setup_inference_logging():
    """Configure logging specifically for inference."""
    # Get logger
    logger = logging.getLogger('vol_project')
    
    # If logger already has handlers, remove them to avoid duplicate logging
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handlers
    log_dir = "inference_logs"  # Can be made into a parameter if needed
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main log file
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'inference_{timestamp}.log')
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger.getChild('inference')

logger = setup_inference_logging()

###############################################################################
# Helper function: Compute sample indices for transformer dataset
###############################################################################
def compute_sample_indices(df: pd.DataFrame, sequence_length: int, sample_every: int) -> List:
    """
    Mimic the sampling logic used in ThreeHeadedTransformerDataset:
      For each (symbol, date) group (sorted by minute), for indices from
      sequence_length to len(group)-1 stepping by sample_every,
      record the original DataFrame index.
    """
    indices = []
    for (sym, d), group in df.groupby(["symbol", "date"], group_keys=False):
        group = group.sort_values("minute")
        n = len(group)
        if n <= sequence_length:
            continue
        # For each valid sequence, record the index corresponding to the prediction point.
        for idx in range(sequence_length, n - 1, sample_every):
            indices.append(group.index[idx])
    return indices

###############################################################################
# 1) Parse command line arguments
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run daily inference for a given model + experiment.\n"
                    "For transformer models, use --dense to generate a prediction for every minute. "
                    "Otherwise, predictions are computed on sampled sequences and then forward-filled."
    )
    parser.add_argument("-m", "--model-type", type=str, required=True,
                        choices=["xgboost", "mlp", "lasso", "seq_mlp", "transformer"],
                        help="Which model to run inference with.")
    parser.add_argument("-e", "--experiment-name", type=str, required=True,
                        help="Name of the experiment folder (e.g. 'top10') under s3://volatility-project/experiments/<model_type>/")
    parser.add_argument("-p", "--run-prefix", type=str, required=True,
                        help="Prefix of the run folder to use (e.g. 'jennifer'). Assumed to be unique within the experiment.")
    parser.add_argument("-d", "--date", type=str, default=None,
                        help="(Optional) Specific date to run inference for (YYYYMMDD). If not set, run for all test dates.")
    parser.add_argument("--dense", action="store_true",
                        help="If set (for transformer models), override sample_every=1 to generate predictions for every minute. "
                             "If not set, predictions will be computed on sparse sequences and then forward-filled.")
    parser.add_argument("-w","--workers", type=int, default=0,
                        help="Number of parallel processes to use for inference (default: 0, meaning no parallelization)")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="If set, skip dates where inference files already exist")
    return parser.parse_args()

###############################################################################
# 2) Find the specific run checkpoint in S3 given model type, experiment name, and run prefix
###############################################################################
def find_checkpoint_with_run_prefix(model_type: str, experiment_name: str, run_prefix: str) -> str:
    fs = s3fs.S3FileSystem(anon=False)
    # Construct the base experiment folder path
    base_dir = f"volatility-project/experiments/{model_type}/{experiment_name}"
    if not fs.exists(base_dir):
        raise FileNotFoundError(f"S3 path s3://{base_dir} does not exist.")
    
    # List all run folders under the base experiment directory.
    all_runs = fs.ls(base_dir)
    # Filter for the folder that starts with the given run_prefix.
    matching_runs = [run for run in all_runs if os.path.basename(run).startswith(run_prefix)]
    if len(matching_runs) == 0:
        raise ValueError(f"No run folder found with prefix '{run_prefix}' under s3://{base_dir}")
    elif len(matching_runs) > 1:
        raise ValueError(f"Multiple run folders found with prefix '{run_prefix}'. Please ensure uniqueness.")
    run_folder = matching_runs[0]
    logger.info(f"Using run folder: {run_folder}")
    
    # Inside the run folder, look for the checkpoint file with appropriate extension
    if model_type == "xgboost":
        checkpoint_path = f"{run_folder}/checkpoints/xgb_model_best.json"
        if not fs.exists(checkpoint_path):
            # Fallback if naming differs
            checkpoint_path = f"{run_folder}/checkpoints/model_best.json"
            if not fs.exists(checkpoint_path):
                raise FileNotFoundError(f"XGBoost checkpoint not found at s3://{checkpoint_path}")
    else:
        checkpoint_path = f"{run_folder}/checkpoints/best_model.pt"
        if not fs.exists(checkpoint_path):
            # Fallback if naming differs
            checkpoint_path = f"{run_folder}/checkpoints/model_best.pt"
            if not fs.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at s3://{checkpoint_path}")
    
    full_checkpoint_path = "s3://" + checkpoint_path
    logger.info(f"Found checkpoint: {full_checkpoint_path}")
    return full_checkpoint_path

###############################################################################
# 3) Loading the model + any needed parameters (and scalers)
###############################################################################
def load_model(checkpoint_s3_path: str, model_type: str, run_folder: str):
    fs = s3fs.S3FileSystem(anon=False)
    
    logger.info(f"Loading model from {checkpoint_s3_path}")
    local_ckpt = "/tmp/model_best.pt"
    fs.get(checkpoint_s3_path.replace("s3://", ""), local_ckpt)
    
    # Load dataset_info.json and model_params.json from config directory
    dataset_info_path = f"{run_folder}/config/dataset_info.json"
    model_params_path = f"{run_folder}/config/model_params.json"
    
    local_dataset_info = "/tmp/dataset_info.json"
    local_model_params = "/tmp/model_params.json"
    
    try:
        fs.get(dataset_info_path.replace("s3://", ""), local_dataset_info)
        fs.get(model_params_path.replace("s3://", ""), local_model_params)
        
        with open(local_dataset_info, 'r') as f:
            dataset_info = json.load(f)
        with open(local_model_params, 'r') as f:
            model_params = json.load(f)
            if "description" in model_params:
                del model_params["description"]
            
        logger.info("Successfully loaded configuration files")
    except Exception as e:
        logger.error(f"Error loading configuration files: {e}")
        raise
    
    feature_count = dataset_info.get('feature_count', len(selected_features))

    # Load scalers if they exist
    scalers_path = f"s3://{run_folder}/checkpoints/scalers.pt"
    local_scalers = "/tmp/scalers.pt"
    feature_scaler = None
    target_scaler = None
    
    fs_path_for_scalers = scalers_path.replace("s3://", "")
    if fs.exists(fs_path_for_scalers):
        fs.get(fs_path_for_scalers, local_scalers)
        try:
            # Add safe globals for numpy scalar types
            torch.serialization.add_safe_globals([
                ('numpy._core.multiarray', 'scalar'),
                ('numpy', 'dtype'),
                ('numpy', 'ndarray'),
                ('sklearn.preprocessing._data', 'StandardScaler')
            ])
            scalers = torch.load(local_scalers, weights_only=False)
            feature_scaler = scalers.get('feature_scaler')
            
            # Handle transformer-specific scaler format
            if scalers.get('is_transformer', False):
                logger.info("Loading transformer-specific scalers...")
                vol_scaler = scalers.get('vol_scaler')
                ret_scaler = scalers.get('ret_scaler')
                target_scaler = (vol_scaler, ret_scaler)
                logger.info(f"Loaded vol_scaler: {type(vol_scaler)}")
                logger.info(f"Loaded ret_scaler: {type(ret_scaler)}")
            else:
                # Handle non-transformer case
                target_scaler = scalers.get('target_scaler')
            
            # Validate scalers
            if feature_scaler is None:
                logger.warning("Feature scaler is None after loading")
            if target_scaler is None:
                logger.warning("Target scaler is None after loading")
            
            logger.info("Loaded feature_scaler and target_scaler from scalers.pt")
            logger.info(f"Target scaler type: {type(target_scaler)}")
            if isinstance(target_scaler, tuple):
                logger.info(f"Target scaler is tuple with types: ({type(target_scaler[0])}, {type(target_scaler[1])})")
            
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}", exc_info=True)
            raise
            
    # Now load the actual model checkpoint with safe globals
    if model_type == "transformer":
        from models.three_headed_transformer import ThreeHeadedTransformer
        assert feature_scaler is not None, "Feature scaler must be provided"
        assert target_scaler is not None, "Target scaler must be provided"
        try:
            # Add safe globals for model loading
            torch.serialization.add_safe_globals([
                ('numpy._core.multiarray', 'scalar'),
                ('numpy', 'dtype'),
                ('numpy', 'ndarray'),
                ('models.three_headed_transformer', 'ThreeHeadedTransformer')
            ])
            checkpoint = torch.load(local_ckpt, map_location=torch.device("cpu"), weights_only=False)
            
            # Get input dimension from dataset_info features
            input_dim = len(dataset_info['features'])
            logger.info(f"Using input dimension {input_dim} from dataset_info features")
            
            # Extract model parameters with defaults if null
            hidden_dim = int(model_params.get('hidden_dim', 128))  # default 128
            nhead = int(model_params.get('nhead', 8))  # default 8
            num_layers = int(model_params.get('num_layers', 4))  # default 4
            dropout = float(model_params.get('dropout', 0.1))  # default 0.1
            max_seq_length = int(model_params['sequence_params']['sequence_length'])
            
            logger.info(f"Using model parameters (with defaults if null):")
            logger.info(f"  hidden_dim: {hidden_dim}")
            logger.info(f"  nhead: {nhead}")
            logger.info(f"  num_layers: {num_layers}")
            logger.info(f"  dropout: {dropout}")
            logger.info(f"  max_seq_length: {max_seq_length}")
            
            # Create model using parameters
            model = ThreeHeadedTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                max_seq_length=max_seq_length
            )
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return model, model_params, feature_scaler, target_scaler, dataset_info
            
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            raise
    
    elif model_type == "xgboost":
        #import xgboost as xgb
        local_ckpt_json = "/tmp/model_best.json"
        fs.get(checkpoint_s3_path.replace("s3://", ""), local_ckpt_json)
        
        booster = xgboost.Booster()
        booster.load_model(local_ckpt_json)
        
        model = xgboost.XGBRegressor()
        model._Booster = booster
        model._le = None
        return model, model_params, feature_scaler, target_scaler, dataset_info
    
    elif model_type == "mlp":
        from models.deep_learning_model import MLPRegressorModel
        checkpoint = torch.load(local_ckpt, map_location=torch.device("cpu"))
        
        hidden_dims = model_params.get('hidden_dims', [64, 32])
        dropout = model_params.get('dropout', 0.2)
        
        model = MLPRegressorModel(
            input_dim=feature_count,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, model_params, feature_scaler, target_scaler, dataset_info
    
    elif model_type == "seq_mlp":
        from models.sequence_model import VolForecastNet
        checkpoint = torch.load(local_ckpt, map_location=torch.device("cpu"))
        
        hidden_dim = model_params.get('hidden_dim', 64)
        dropout = model_params.get('dropout', 0.2)
        
        short_in_dim = feature_count
        long_in_dim = feature_count
        
        model = VolForecastNet(
            short_in_dim=short_in_dim,
            long_in_dim=long_in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, model_params, feature_scaler, target_scaler, dataset_info
    
    elif model_type == "lasso":
        from models.lasso_model import AsymmetricLasso
        checkpoint = torch.load(local_ckpt, map_location=torch.device("cpu"), weights_only=True)
        
        model_dict = checkpoint["model_state_dict"]
        model = AsymmetricLasso(
            alpha=checkpoint.get("alpha", 1.0),
            over_penalty=checkpoint.get("over_penalty", 1.0),
            under_penalty=checkpoint.get("under_penalty", 1.0),
            max_iter=checkpoint.get("max_iter", 1000),
            tol=checkpoint.get("tol", 1e-4)
        )
        model.coef_ = model_dict["coef_"]
        model.intercept_ = model_dict["intercept_"]
        model.feature_names_ = model_dict["feature_names_"]
        
        model.eval_ = False
        return model, model_params, feature_scaler, target_scaler, dataset_info
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

###############################################################################
# 4) Determine which dates to run
###############################################################################
def get_dates_to_infer(single_date: Optional[str] = None):
    """
    Return a list of dates (as strings YYYYMMDD) to run inference on.
    If single_date is provided, return [single_date] only.
    Otherwise, return NYSE trading days between 2024-08-01 and 2025-01-14.
    """
    if single_date:
        return [single_date]

    from pandas_market_calendars import get_calendar
    nyse = get_calendar('NYSE')
    
    start_date = "20240116"
    end_date = "20250114"
    
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.strftime('%Y%m%d').tolist()
    return trading_days

###############################################################################
# 5) Load daily Parquet
###############################################################################
def load_parquet_for_inference(date_str: str) -> pd.DataFrame:
    fs = s3fs.S3FileSystem(anon=False)
    file_path = f"volatility-project/data/features/attention_df/all/{date_str}.parquet"
    
    if not fs.exists(file_path):
        logger.warning(f"File not found: s3://{file_path}")
        return pd.DataFrame()
    
    with fs.open(file_path, "rb") as f:
        table = pq.read_table(f)
        df = table.to_pandas()
        df['date'] = pd.to_datetime(date_str, format='%Y%m%d')
    # Filter for top 1000 universe
    df = df[df['in_top1000_universe'] == 1]
    return df

###############################################################################
# 6) Preprocess the data to match training
###############################################################################
def preprocess_inference_data(df: pd.DataFrame, model_type: str, feature_scaler=None):
    """
    Replicate training-time preprocessing steps.
    1) Replace infinities with NaN
    2) Handle missing values differently for XGBoost vs other models
    3) (Optional) Feature scaling for non-XGBoost models
    """
    if df.empty:
        return df
    
    # Replace infinities with NaN for all models
    df[selected_features] = df[selected_features].replace([np.inf, -np.inf], np.nan)
    
    if model_type == "xgboost":
        logger.info("Using XGBoost - keeping NaN values as-is")
        return df
    
    # For non-XGBoost models, proceed with original imputation logic
    df = handle_missing_values(df, feature_cols=selected_features)
    
    # Apply feature scaling for non-XGBoost models if scaler provided
    if feature_scaler is not None:
        logger.info("Applying feature_scaler to inference data...")
        # Ensure features are in the same order as during training
        feature_df = df[selected_features].copy()
        
        # Log feature order for debugging
        logger.info("Feature order for scaling:")
        for i, feat in enumerate(selected_features):
            logger.info(f"{i}: {feat}")
        
        # Transform features in the correct order
        scaled_features = feature_scaler.transform(feature_df)
        
        # Assign back to DataFrame maintaining column names
        df[selected_features] = pd.DataFrame(
            scaled_features, 
            columns=selected_features,
            index=df.index
        )
    
    return df

###############################################################################
# 7) Inference for each model
###############################################################################
def predict_day(
    df: pd.DataFrame,
    model: Any,
    model_type: str,
    model_params: Dict,
    dataset_info: Dict,
    feature_scaler: Optional[Any] = None,
    target_scaler: Optional[Any] = None,
    dense: bool = False
) -> pd.DataFrame:
    """Generate predictions for a single day."""
    logger.info(f"\nGenerating predictions for model type: {model_type}")
    
    # Check for CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Preprocess data
    df = preprocess_inference_data(df, model_type, feature_scaler)
    if df.empty:
        logger.warning("No data after preprocessing")
        return df
    
    if model_type == "transformer":
        try:
            from data_pipeline.sequence_dataset import ThreeHeadedTransformerDataset
            logger.info("Creating transformer dataset...")
            
            # Get sequence parameters from model_params
            sequence_params = model_params['sequence_params']
            # If dense mode is requested, override sample_every to 1.
            if dense:
                logger.info("Dense inference selected: overriding sample_every to 1")
                sample_every_val = 1
            else:
                sample_every_val = sequence_params.get('sample_every', 10)
            
            # Create dataset with the desired parameters
            dataset = ThreeHeadedTransformerDataset(
                df=df,
                feature_cols=dataset_info['features'],
                sequence_length=sequence_params['sequence_length'],
                sample_every=sample_every_val
            )
            
            # Create dataloader with appropriate batch size for GPU
            batch_size = model_params.get('batch_size', 8192)  # Default to large batch size for inference
            if use_cuda:
                batch_size *= 2  # Double batch size for GPU
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Run data loading in the same process
                pin_memory=use_cuda  # Enable pin_memory for faster GPU transfer
            )
            
            # Move model to GPU if available
            model = model.to(device)
            
            t0 = time.time()
            # Get predictions using the transformer model's predict function
            with torch.cuda.amp.autocast(enabled=use_cuda):  # Enable automatic mixed precision
                predictions = predict_three_headed_model(
                    model=model,
                    dataloader=dataloader,
                    target_scaler=target_scaler,
                    use_cuda=use_cuda
                )
            
            # Ensure predictions are 1D before applying target_scaler
            if target_scaler is not None:
                for key in predictions:
                    if isinstance(predictions[key], np.ndarray) and predictions[key].ndim > 1:
                        predictions[key] = predictions[key].ravel()
                    if isinstance(predictions[key], list):
                        predictions[key] = np.array(predictions[key]).ravel()
            
            latency = time.time() - t0
            logger.info(f"Transformer inference latency: {latency:.2f} seconds")
            
            # Now, if dense mode was used, we assume that predictions nearly align with
            # the original DataFrame (except for the first sequence_length rows).
            # Otherwise, we need to merge sparse predictions back via computed indices and forward-fill.
            seq_len = sequence_params['sequence_length']
            if not dense:
                # Compute original indices for each sequence sample using the same grouping logic.
                seq_indices = compute_sample_indices(df, seq_len, sample_every_val)
                if len(seq_indices) != len(predictions['volatility']):
                    raise ValueError(f"Mismatch in computed sample indices length ({len(seq_indices)}) and predictions length ({len(predictions['volatility'])})")
                
                # Create a Series for each prediction type and merge with original DataFrame.
                vol_series = pd.Series(predictions['volatility'], index=seq_indices)
                ret_series = pd.Series(predictions['returns'], index=seq_indices)
                vol_conf_series = pd.Series(predictions['vol_confidence'], index=seq_indices)
                ret_conf_series = pd.Series(predictions['ret_confidence'], index=seq_indices)
                
                full_pred = df.copy()
                # Initialize prediction columns with NaN
                full_pred["predicted.volatility"] = np.nan
                full_pred["predicted.returns"] = np.nan
                full_pred["predicted.vol_confidence"] = np.nan
                full_pred["predicted.ret_confidence"] = np.nan
                
                # Assign predictions at computed indices
                full_pred.loc[vol_series.index, "predicted.volatility"] = vol_series
                full_pred.loc[ret_series.index, "predicted.returns"] = ret_series
                full_pred.loc[vol_conf_series.index, "predicted.vol_confidence"] = vol_conf_series
                full_pred.loc[ret_conf_series.index, "predicted.ret_confidence"] = ret_conf_series
                
                # Forward fill within each symbol group
                full_pred["predicted.volatility"] = full_pred.groupby("symbol")["predicted.volatility"].ffill()
                full_pred["predicted.returns"] = full_pred.groupby("symbol")["predicted.returns"].ffill()
                full_pred["predicted.vol_confidence"] = full_pred.groupby("symbol")["predicted.vol_confidence"].ffill()
                full_pred["predicted.ret_confidence"] = full_pred.groupby("symbol")["predicted.ret_confidence"].ffill()
                
                return full_pred
            else:
                # Dense mode: compute sample indices for alignment
                seq_indices = compute_sample_indices(df, seq_len, 1)
                if len(seq_indices) != len(predictions['volatility']):
                    raise ValueError(f"Mismatch in dense mode: computed sample indices length ({len(seq_indices)}) vs predictions length ({len(predictions['volatility'])})")
                
                pred_df = df.copy()
                # Initialize prediction columns with NaN
                pred_df["predicted.volatility"] = np.nan
                pred_df["predicted.returns"] = np.nan
                pred_df["predicted.vol_confidence"] = np.nan
                pred_df["predicted.ret_confidence"] = np.nan
                
                # Assign predictions at computed indices
                pred_df.loc[seq_indices, "predicted.volatility"] = predictions['volatility']
                pred_df.loc[seq_indices, "predicted.returns"] = predictions['returns']
                pred_df.loc[seq_indices, "predicted.vol_confidence"] = predictions['vol_confidence']
                pred_df.loc[seq_indices, "predicted.ret_confidence"] = predictions['ret_confidence']
                
                # Forward fill within each symbol group
                pred_df["predicted.volatility"] = pred_df.groupby("symbol")["predicted.volatility"].ffill()
                pred_df["predicted.returns"] = pred_df.groupby("symbol")["predicted.returns"].ffill()
                pred_df["predicted.vol_confidence"] = pred_df.groupby("symbol")["predicted.vol_confidence"].ffill()
                pred_df["predicted.ret_confidence"] = pred_df.groupby("symbol")["predicted.ret_confidence"].ffill()
                
                return pred_df
            
        except Exception as e:
            logger.error(f"Failed to create transformer dataset: {str(e)}")
            raise
    
    elif model_type == "xgboost":
        try:
            t0 = time.time()
            # Create a small dataset to use the same scaling logic
            temp_dataset = VolatilityDataset(
                df=df,
                feature_cols=selected_features,
                target_col="Y_log_vol_10min_lag_1m",  # Default to volatility target
                target_scaler=target_scaler,
                fit_target=False
            )
            X_data = temp_dataset.X.numpy()
            
            # Handle both XGBRegressor and Booster models
            if isinstance(model, XGBRegressor):
                preds_scaled = model.predict(X_data)
            else:  # Booster model from xgboost.train()
                dmatrix = xgboost.DMatrix(X_data)
                preds_scaled = model.predict(dmatrix)
            
            # Invert StandardScaler if used
            if target_scaler is not None:
                preds_unscaled = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()
            else:
                preds_unscaled = preds_scaled
                
            # Invert log1p transformation
            final_preds = np.expm1(preds_unscaled)
            
            pred_df = df.copy()
            pred_df["predicted.volatility"] = final_preds
            
            latency = time.time() - t0
            logger.info(f"XGBoost inference latency: {latency:.2f} seconds")
            return pred_df
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            raise
            
    elif model_type == "mlp":
        try:
            from models.deep_learning_model import get_predictions
            t0 = time.time()
            # Move model to GPU if available
            model = model.to(device)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                pred_df = get_predictions(
                    model=model,
                    df=df,
                    feature_cols=selected_features,
                    target_scaler=target_scaler,
                    use_cuda=use_cuda,
                    batch_size=8192 * (2 if use_cuda else 1)  # Larger batches for GPU
                )
            # Rename the prediction column
            pred_df = pred_df.rename(columns={f"{model_type}.predict": "predicted.volatility"})
            latency = time.time() - t0
            logger.info(f"MLP inference latency: {latency:.2f} seconds")
            return pred_df
        except Exception as e:
            logger.error(f"MLP prediction failed: {e}")
            raise
            
    elif model_type == "seq_mlp":
        try:
            sequence_length = model_params.get("sequence_length", 30)
            batch_size = model_params.get("batch_size", 8192)
            if use_cuda:
                batch_size *= 2  # Double batch size for GPU
            
            dataset = VolatilityDataset(
                df,
                feature_cols=selected_features,
                sequence_length=sequence_length,
                feature_scaler=None,
                target_scaler=None,
                training=False
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=use_cuda
            )
            
            # Move model to GPU if available
            model = model.to(device)
            model.eval()
            
            all_preds = []
            t0 = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_cuda):
                for batch in dataloader:
                    inputs = batch[0].to(device, non_blocking=True)
                    outputs = model(inputs)
                    if target_scaler:
                        outputs = target_scaler.inverse_transform(outputs.cpu().numpy())
                    else:
                        outputs = outputs.cpu().numpy()
                    all_preds.extend(outputs)
                    
            latency = time.time() - t0
            logger.info(f"Sequence MLP inference latency: {latency:.2f} seconds")
            pred_df = df.copy()
            pred_df["predicted.volatility"] = all_preds
            return pred_df
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("CUDA out of memory. Try reducing batch size.")
                raise
            logger.error(f"Sequence MLP prediction failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Sequence MLP prediction failed: {e}")
            raise
            
    elif model_type == "lasso":
        try:
            t0 = time.time()
            pred_df = predict_lasso_model(
                model=model,
                df=df,
                feature_cols=selected_features,
                target_scaler=target_scaler
            )
            # Rename the prediction column
            pred_df = pred_df.rename(columns={f"{model_type}.predict": "predicted.volatility"})
            latency = time.time() - t0
            logger.info(f"Lasso inference latency: {latency:.2f} seconds")
            return pred_df
        except Exception as e:
            logger.error(f"Lasso prediction failed: {e}")
            raise
            
    else:
        raise ValueError(f"Unknown model type: {model_type}")

###############################################################################
# 8) Save results (CSV) + Debug Logs
###############################################################################
def save_inference_results(df: pd.DataFrame, model_type: str, experiment_name: str,
                           run_folder: str, date_str: str, dense: bool = False):
    from inference.keep_cols import KEEP_COLUMNS
    
    # Get prediction columns (those starting with 'predicted.')
    pred_cols = [col for col in df.columns if col.startswith('predicted.')]
    
    # Keep only specified columns and prediction columns
    cols_to_keep = KEEP_COLUMNS + pred_cols
    df_to_save = df[cols_to_keep].copy()
    
    fs = s3fs.S3FileSystem(anon=False)
    # Create an inference folder under the run folder (if it doesn't exist)
    inference_folder = f"{run_folder}/inference"
    if not fs.exists(inference_folder.replace("s3://", "")):
        fs.makedirs(inference_folder.replace("s3://", ""))
    
    # Save original format file (dense or sparse)
    suffix = ".dense_inference.csv" if dense else ".inference.csv"
    out_path = f"{inference_folder}/{date_str}{suffix}"
    s3_dest = out_path if out_path.startswith("s3://") else "s3://" + out_path
    logger.info(f"\nWriting inference results to {s3_dest}")
    with fs.open(out_path.replace("s3://", ""), "w") as f:
        df_to_save.to_csv(f, index=False)
    
    # Create and save 5-minute sampled version
    # Filter for minutes ending in :00, :05, :10, etc
    df_5min = df_to_save[df_to_save['minute'].str[-2:].isin(['00', '05', '10', '15', '20', 
                                                             '25', '30', '35', '40', '45', 
                                                             '50', '55'])].copy()
    five_min_path = f"{inference_folder}/{date_str}.5min.csv"
    s3_5min_dest = five_min_path if five_min_path.startswith("s3://") else "s3://" + five_min_path
    logger.info(f"Writing 5-minute sampled inference results to {s3_5min_dest}")
    with fs.open(five_min_path.replace("s3://", ""), "w") as f:
        df_5min.to_csv(f, index=False)
    
    logger.info(f"Saved {len(pred_cols)} prediction columns and {len(KEEP_COLUMNS)} keep columns")
    logger.info(f"Full dataset: {len(df_to_save)} rows, 5-min sampled: {len(df_5min)} rows")

###############################################################################
# MAIN inference loop
###############################################################################
def process_single_date(args_tuple):
    """
    Process a single date with the given parameters.
    Args tuple contains: (date_str, model, model_type, model_params, dataset_info, 
                         feature_scaler, target_scaler, dense, run_folder, experiment_name, no_overwrite)
    """
    (date_str, model, model_type, model_params, dataset_info, 
     feature_scaler, target_scaler, dense, run_folder, experiment_name, no_overwrite) = args_tuple
    
    # Use a single logger without re-initializing
    logger = logging.getLogger('vol_project.inference')
    
    # Check if inference files already exist
    fs = s3fs.S3FileSystem(anon=False)
    inference_folder = f"{run_folder}/inference"
    suffix = ".dense_inference.csv" if dense else ".inference.csv"
    main_path = f"{inference_folder}/{date_str}{suffix}"
    five_min_path = f"{inference_folder}/{date_str}.5min.csv"
    
    if no_overwrite and fs.exists(main_path.replace("s3://", "")) and fs.exists(five_min_path.replace("s3://", "")):
        logger.info(f"Inference files already exist for {date_str}, skipping (--no-overwrite is set)")
        return date_str, True, 0
    
    logger.info(f"Processing date: {date_str}")
    
    start_t = time.time()
    
    # Load data
    df_day = load_parquet_for_inference(date_str)
    if df_day.empty:
        logger.warning(f"No data found for {date_str}, skipping.")
        return date_str, False, 0
    
    # Predict
    try:
        pred_df = predict_day(
            df_day, model, model_type, model_params,
            dataset_info=dataset_info,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            dense=dense
        )
    except Exception as e:
        logger.error(f"Prediction failed for {date_str}: {e}")
        return date_str, False, 0
    
    # Save
    try:
        save_inference_results(
            pred_df, 
            model_type,
            experiment_name,
            run_folder, 
            date_str,
            dense=dense
        )
    except Exception as e:
        logger.error(f"Failed to save inference results for {date_str}: {e}")
        return date_str, False, 0
    
    elapsed = time.time() - start_t
    logger.info(f"Completed inference for {date_str} in {elapsed:.2f} seconds.")
    return date_str, True, elapsed

def main():
    args = parse_arguments()
    logger.info(f"Running inference for model={args.model_type}, experiment={args.experiment_name}, "
                f"run prefix={args.run_prefix}, date={args.date}, dense={args.dense}, "
                f"parallel processes={args.workers}, no_overwrite={args.no_overwrite}")
    
    # 1. Find the specific run checkpoint using the provided run prefix.
    checkpoint_path = find_checkpoint_with_run_prefix(args.model_type, args.experiment_name, args.run_prefix)
    run_folder = os.path.dirname(os.path.dirname(checkpoint_path))
    
    # 2. Load model + scalers + dataset_info
    model, model_params, feature_scaler, target_scaler, dataset_info = load_model(
        checkpoint_path, args.model_type, run_folder
    )
    logger.info("Model and scalers loaded successfully.")
    
    # 3. Determine which dates to run
    dates_to_run = get_dates_to_infer(args.date)
    logger.info(f"Will process {len(dates_to_run)} dates using {args.workers} parallel processes")
    
    # Prepare arguments for parallel processing
    process_args = [
        (date_str, model, args.model_type, model_params, dataset_info, 
         feature_scaler, target_scaler, args.dense, run_folder, args.experiment_name, args.no_overwrite)
        for date_str in dates_to_run
    ]
    
    # Run either in parallel or sequential mode based on workers argument
    if args.workers > 0:
        logger.info(f"Running in parallel mode with {args.workers} workers")
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            results = pool.map(process_single_date, process_args)
    else:
        logger.info("Running in sequential mode")
        results = [process_single_date(args) for args in process_args]
    
    # Process results
    successful_dates = []
    failed_dates = []
    total_time = 0.0
    
    for date_str, success, elapsed in results:
        if success:
            successful_dates.append(date_str)
            total_time += elapsed
        else:
            failed_dates.append(date_str)
    
    # Print summary
    logger.info("\n=== Inference Summary ===")
    logger.info(f"Total dates processed: {len(dates_to_run)}")
    logger.info(f"Successful: {len(successful_dates)}")
    logger.info(f"Failed: {len(failed_dates)}")
    if failed_dates:
        logger.info("Failed dates: " + ", ".join(failed_dates))
    if successful_dates:
        avg_latency = total_time / len(successful_dates)
        logger.info(f"Average inference latency per date: {avg_latency:.2f} seconds")
    logger.info("All done!")

if __name__ == "__main__":
    main()