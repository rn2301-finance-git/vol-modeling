import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import s3fs
import torch
from torch.utils.data import DataLoader
from evaluation.run_manager import RunManager
import logging
from scipy import stats
import scipy.stats

class ModelEvaluator:
    """Evaluator that can handle both regular and sequence-based models."""
    
    def __init__(self, 
                 mode: str = "subset",
                 over_penalty: float = 2.0, 
                 under_penalty: float = 1.0,
                 run_manager: Optional[RunManager] = None):
        """Initialize evaluator with mode and penalty parameters."""
        if mode not in ["full", "subset"]:
            raise ValueError("Mode must be either 'full' or 'subset'")
            
        self.mode = mode
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty
        self.run_manager = run_manager
        self.results_log = []
        self.s3_bucket = "volatility-project"
        self.fs = s3fs.S3FileSystem(anon=False)
        
        # Setup logging
        self.logger = logging.getLogger('model_training')
        
        # Create evaluations directory in S3 if it doesn't exist
        self.base_path = f"s3://{self.s3_bucket}/evaluations"
        try:
            self.fs.mkdir(self.base_path, create_parents=True)
        except Exception as e:
            self.logger.warning(f"Error creating evaluations directory: {e}")
        
    def asymmetric_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate asymmetric MSE with higher penalty for over-prediction.
        Handles edge cases where all predictions are over or under.
        """
        diff = y_pred - y_true
        over_mask = diff > 0
        under_mask = ~over_mask
        
        # Get the differences for each case
        over_diff = diff[over_mask]
        under_diff = diff[under_mask]
        
        # Calculate errors, defaulting to 0 if no samples in that category
        over_error = self.over_penalty * np.mean(over_diff**2) if len(over_diff) > 0 else 0.0
        under_error = self.under_penalty * np.mean(under_diff**2) if len(under_diff) > 0 else 0.0
        
        # If we have samples in both categories, return average
        # If only one category has samples, return that error
        # If no errors (perfect prediction), return 0
        if len(over_diff) > 0 and len(under_diff) > 0:
            return (over_error + under_error) / 2
        elif len(over_diff) > 0:
            return over_error
        elif len(under_diff) > 0:
            return under_error
        else:
            return 0.0  # Perfect prediction case

    def evaluate_predictions(self, 
                           predictions: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
                           y_true_col: str = 'Y_log_vol_10min_lag_1m',
                           y_pred_col: str = 'predicted_vol',
                           group_cols: Optional[List[str]] = None) -> Dict:
        """
        Evaluate predictions with multiple metrics.
        
        Parameters:
        -----------
        predictions : Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]
            Either:
            - DataFrame with predictions and true values
            - Tuple of (y_true, y_pred) arrays for sequence models
        y_true_col : str
            Name of true values column (only used if predictions is DataFrame)
        y_pred_col : str
            Name of predicted values column (only used if predictions is DataFrame)
        group_cols : Optional[List[str]]
            Optional list of columns to group by (only used if predictions is DataFrame)
        """
        # Handle different input types
        if isinstance(predictions, pd.DataFrame):
            y_true = predictions[y_true_col].values
            y_pred = predictions[y_pred_col].values
            df = predictions
        else:
            y_true, y_pred = predictions
            df = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred
            })
        
        # Base metrics
        metrics = {
            'rank_corr': spearmanr(y_true, y_pred)[0],
            'asymm_mse': self.asymmetric_mse(y_true, y_pred),
            'rmse': np.sqrt(np.mean((y_true - y_pred)**2)),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'sample_count': len(y_true)
        }
        
        # Add symbol count if available
        if isinstance(predictions, pd.DataFrame) and 'symbol' in predictions.columns:
            metrics['unique_symbols'] = predictions['symbol'].nunique()
            
            # Time of day analysis (if applicable)
            if all(col in predictions.columns for col in ['minute_num', 'hour']):
                open_mask = ((df['hour'] == 9) & (df['minute_num'] >= 35)) | \
                           ((df['hour'] == 10) & (df['minute_num'] <= 30))
                metrics['open_rank_corr'] = spearmanr(
                    df[open_mask][y_true_col], 
                    df[open_mask][y_pred_col]
                )[0]
                metrics['open_asymm_mse'] = self.asymmetric_mse(
                    df[open_mask][y_true_col].values,
                    df[open_mask][y_pred_col].values
                )
        
        return metrics

    def evaluate_sequence_model(self,
                              model: torch.nn.Module,
                              data_loader: DataLoader,
                              device: torch.device) -> Dict:
        """
        Evaluate a sequence model using a DataLoader.
        
        Parameters:
        -----------
        model : torch.nn.Module
            The sequence model to evaluate
        data_loader : DataLoader
            DataLoader containing the evaluation data
        device : torch.device
            Device to run the model on
        """
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_short, X_long, y in data_loader:
                X_short = X_short.to(device)
                X_long = X_long.to(device)
                preds = model(X_short, X_long).cpu().numpy()
                targets = y.numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        return self.evaluate_predictions(
            predictions=(np.array(all_targets), np.array(all_preds))
        )

    def evaluate_three_headed_model(self,
                                  model: torch.nn.Module,
                                  data_loader: DataLoader,
                                  device: torch.device) -> Dict:
        """
        Evaluate a three-headed transformer model using a DataLoader.
        
        The model is expected to output a dictionary with keys:
          - 'returns'
          - 'volatility'
          - 'vol_logvar'
          - 'ret_confidence'
        
        The DataLoader should yield batches in the form:
            (X, targets_dict)
        where:
            - X is the input sequence tensor
            - targets_dict contains 'returns' and 'volatility' tensors
        
        Returns:
        --------
        Dict
            A dictionary containing evaluation metrics computed separately for volatility and returns.
        """
        model.eval()
        all_preds_vol = []
        all_preds_ret = []
        all_targets_vol = []
        all_targets_ret = []
        
        with torch.no_grad():
            for X, targets in data_loader:
                X = X.to(device)
                targets_vol = targets['volatility'].to(device)
                targets_ret = targets['returns'].to(device)
                
                outputs = model(X)
                
                # For evaluation, we use the direct outputs from the corresponding heads
                preds_vol = outputs['volatility'].cpu().numpy()
                preds_ret = outputs['returns'].cpu().numpy()
                
                all_preds_vol.extend(preds_vol)
                all_preds_ret.extend(preds_ret)
                all_targets_vol.extend(targets_vol.cpu().numpy())
                all_targets_ret.extend(targets_ret.cpu().numpy())
        
        # Convert lists to numpy arrays
        all_preds_vol = np.array(all_preds_vol).reshape(-1)
        all_preds_ret = np.array(all_preds_ret).reshape(-1)
        all_targets_vol = np.array(all_targets_vol).reshape(-1)
        all_targets_ret = np.array(all_targets_ret).reshape(-1)
        
        # Evaluate predictions for volatility and returns separately
        metrics_vol = self.evaluate_raw_arrays(all_preds_vol, all_targets_vol)
        metrics_ret = self.evaluate_raw_arrays(all_preds_ret, all_targets_ret)
        
        # Calculate asymmetric MSE for volatility predictions
        vol_asymm_mse = self.asymmetric_mse(all_targets_vol, all_preds_vol)
        
        # Combine the evaluation metrics into one dictionary
        combined_metrics = {
            "volatility_rmse": metrics_vol["rmse"],
            "volatility_rank_corr": metrics_vol["rank_corr"],
            "volatility_r2": metrics_vol["r2"],
            "volatility_mae": metrics_vol["mae"],
            "volatility_asymm_mse": vol_asymm_mse,
            "returns_rmse": metrics_ret["rmse"],
            "returns_rank_corr": metrics_ret["rank_corr"],
            "returns_r2": metrics_ret["r2"],
            "returns_mae": metrics_ret["mae"],
            "sample_count": len(all_targets_vol)
        }
        
        return combined_metrics

    def convert_floats(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_floats(item) for item in obj]
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return obj

    def log_results(self, 
                   model_name: str,
                   metrics: Dict,
                   params: Optional[Dict] = None,
                   notes: str = "",
                   dataset_type: str = "train") -> None:
        """Log evaluation results."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'parameters': params or {},
            'notes': notes,
            'dataset_type': dataset_type
        }
        
        # Log to run manager if available
        if self.run_manager:
            self.run_manager.log_metrics(
                metrics=metrics,
                step=len(self.results_log),
                prefix=f'Evaluation/{dataset_type}'
            )
        
        # Append to results log
        self.results_log.append(result)


    def get_summary(self, include_historical: bool = True) -> pd.DataFrame:
        """
        Return a DataFrame summarizing evaluations.
        
        Parameters:
        -----------
        include_historical : bool
            If True, attempt to load and include historical results from S3
        """
        # Start with current results
        all_results = self.results_log.copy()  # Make a copy to avoid modifying the original
        
        if include_historical:
            # Load historical results from S3 based on mode
            mode_prefix = "subset" if self.mode == "subset" else "full"
            s3_pattern = f"s3://{self.s3_bucket}/evaluations/{mode_prefix}/**/*.json"
            
            fs = s3fs.S3FileSystem(anon=False)
            all_files = fs.glob(s3_pattern)
            
            # Load historical results without modifying self.results_log
            historical_results = []
            for file_path in all_files:
                with fs.open(file_path, 'r') as f:
                    file_results = json.load(f)
                    if isinstance(file_results, list):
                        historical_results.extend(file_results)
            
            # Combine current and historical results
            all_results.extend(historical_results)
        
        summary_data = []
        for result in all_results:
            row = {
                'model_name': result['model_name'],
                'timestamp': result['timestamp'],
                'evaluation_mode': result.get('evaluation_mode', self.mode),
                **result['metrics']
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

    def print_latest_summary(self, n_latest: int = 5) -> None:
        """Print a formatted summary of the n most recent evaluations."""
        summary = self.get_summary()
        latest = summary.sort_values('timestamp', ascending=False).head(n_latest)
        
        print(f"\n{'='*80}")
        print(f"Latest {n_latest} evaluations for mode: {self.mode}")
        print(f"{'='*80}")
        
        for _, row in latest.iterrows():
            print(f"\nModel: {row['model_name']}")
            print(f"Timestamp: {row['timestamp']}")
            print(f"Rank Correlation: {row['rank_corr']:.4f}")
            print(f"Asymmetric MSE: {row['asymm_mse']:.6f}")
            
            # Only print open period metrics if they exist
            if 'open_rank_corr' in row:
                print(f"Open Period Rank Correlation: {row['open_rank_corr']:.4f}")
            
            print(f"Sample Count: {row['sample_count']:,}")
            print(f"{'='*40}")

    def evaluate_raw_arrays(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics from raw numpy arrays of predictions and targets.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            True target values
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        # Ensure predictions and targets are 1D arrays
        predictions = np.squeeze(predictions)
        targets = np.squeeze(targets)
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Compute RMSE
        metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # Compute rank correlation (Spearman's rho)
        metrics['rank_corr'] = scipy.stats.spearmanr(predictions, targets)[0]
        
        # Add other metrics as needed
        metrics['mae'] = np.mean(np.abs(predictions - targets))
        metrics['mse'] = np.mean((predictions - targets) ** 2)
        
        # Handle NaN values
        for key in metrics:
            if np.isnan(metrics[key]):
                self.logger.warning(f"NaN value detected for {key}")
                metrics[key] = float('inf') if key in ['rmse', 'mae', 'mse'] else 0.0
        
        return metrics