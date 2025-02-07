import os
import json
import logging
import names
import tempfile
from datetime import datetime
import s3fs
import shutil
from typing import Dict, Any, Optional
import torch
import numpy as np
import sys
import io

def setup_logging(test_mode: bool = False, log_dir: str = None) -> logging.Logger:
    """Configure logging specifically for inference."""
    # Get logger
    logger = logging.getLogger('BAM')
    
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
    
    # StringIO handler for buffer
    string_io_handler = logging.StreamHandler(RunManager.log_buffer)
    string_io_handler.setLevel(logging.INFO)
    string_io_handler.setFormatter(file_formatter)
    logger.addHandler(string_io_handler)
    
    # File handlers
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main log file - use 'test' or 'prod' based on test_mode
        mode_str = 'test' if test_mode else 'prod'
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'bam_{mode_str}_{timestamp}.log')
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Initial log messages
    logger.info("\n" + "="*80)
    logger.info(f"Starting new run at {datetime.now()}")
    if test_mode:
        logger.info("RUNNING IN TEST MODE")
    logger.info("="*80 + "\n")
    
    return logger

class RunManager:
    """Manages experiment runs, including logging and checkpoints."""
    
    # Class-level log buffer shared across instances
    log_buffer = io.StringIO()
    
    def __init__(self, 
                 model_type: str,
                 mode: str,
                 s3_bucket: str = "bam-volatility-project",
                 base_path: str = "experiments",
                 experiment_name: Optional[str] = None):
        """
        Initialize a new run manager.
        
        Parameters:
        -----------
        model_type : str
            Type of model being trained (e.g., 'mlp', 'seq_mlp')
        mode : str
            Training mode ('full' or 'subset')
        s3_bucket : str
            S3 bucket for storing run data
        base_path : str
            Base path within the bucket for storing experiments
        experiment_name : str, optional
            Name of the experiment group. If None, defaults to model_type
        """
        self.model_type = model_type
        self.mode = mode
        self.s3_bucket = s3_bucket
        self.experiment_name = experiment_name or model_type
        
        # Standardize the path structure
        self.base_path = base_path.rstrip('/')
        self.run_id = self._generate_run_id()
        
        # Create a consistent path structure:
        # s3://bucket/base_path/experiment_name/run_id/
        self.run_dir = f"s3://{s3_bucket}/{self.base_path}/{self.experiment_name}/{self.run_id}"
        
        # Initialize S3 filesystem
        self.fs = s3fs.S3FileSystem(anon=False)
        
        # Create temp directory for local files
        self.temp_dir = tempfile.mkdtemp()
        self.local_checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        os.makedirs(self.local_checkpoint_dir, exist_ok=True)
        
        # Clear the class-level log buffer at the start of each run
        RunManager.log_buffer.truncate(0)
        RunManager.log_buffer.seek(0)
        
        # Setup logging with the shared buffer
        self.logger = setup_logging(
            test_mode=(mode == "test"),
            log_dir=os.path.join(self.temp_dir, 'logs')
        )
        
        # Setup directory structure
        self._setup_directories()
        
        # Store best model info
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        self.logger.info(f"Initialized new run: {self.run_id}")
        self.logger.info(f"Local checkpoint directory: {self.local_checkpoint_dir}")
        self.logger.info(f"S3 base path: {self.run_dir}")
        
    def _generate_run_id(self) -> str:
        """Generate a unique run ID using creative name and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        creative_name = names.get_first_name().lower()
        return f"{creative_name}_{self.model_type}_{timestamp}"
    
    def _setup_directories(self):
        """Create necessary directories in S3."""
        directories = ['config', 'checkpoints', 'metrics', 'logs']
        for dir_name in directories:
            path = f"{self.run_dir}/{dir_name}"
            if not self.fs.exists(path.replace('s3://', '')):
                self.fs.makedirs(path.replace('s3://', ''))
    
    def save_config(self, 
                   args: Dict[str, Any],
                   dataset_info: Dict[str, Any],
                   model_params: Dict[str, Any]):
        """Save configuration information."""
        configs = {
            'args.json': args,
            'dataset_info.json': dataset_info,
            'model_params.json': model_params
        }
        
        for filename, data in configs.items():
            path = f"{self.run_dir}/config/{filename}"
            with self.fs.open(path.replace('s3://', ''), 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {filename}")
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_metrics: dict,
        is_best: bool = False,
        feature_scaler = None,
        target_scaler = None
    ):
        """Save model checkpoint and scalers."""
        checkpoint_path = os.path.join(self.local_checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        best_model_path = os.path.join(self.local_checkpoint_dir, 'best_model.pt')
        scalers_path = os.path.join(self.local_checkpoint_dir, 'scalers.pt')

        # Handle both PyTorch models and scikit-learn style models
        if hasattr(model, 'state_dict'):
            # PyTorch model
            model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
        else:
            # Scikit-learn style model (already a dict or has attributes)
            model_state = {
                'model_state_dict': model if isinstance(model, dict) else {
                    'coef_': getattr(model, 'coef_', None),
                    'intercept_': getattr(model, 'intercept_', None),
                    'feature_names_': getattr(model, 'feature_names_', None)
                },
                'optimizer_state_dict': None,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }

        # Save scalers if provided
        if feature_scaler is not None or target_scaler is not None:
            # Handle transformer case where target_scaler is a tuple
            if isinstance(target_scaler, tuple):
                self.logger.info(f"Target scaler is a tuple: {target_scaler}")
                vol_scaler, ret_scaler = target_scaler
                scalers_state = {
                    'feature_scaler': feature_scaler,
                    'vol_scaler': vol_scaler,
                    'ret_scaler': ret_scaler,
                    'is_transformer': True
                }
            else:
                self.logger.info(f"Target scaler is not a tuple: {target_scaler}")
                scalers_state = {
                    'feature_scaler': feature_scaler,
                    'target_scaler': target_scaler,
                    'is_transformer': False
                }
            
            torch.save(scalers_state, scalers_path)
            self.logger.info(f"Saved scalers to {scalers_path} "
                           f"(target_scaler is tuple: {isinstance(target_scaler, tuple)}, "
                           f"feature_scaler is None: {feature_scaler is None}, "
                           f"target_scaler is None: {target_scaler is None})")

            # Add scaler paths to model state for reference
            model_state['scalers_path'] = scalers_path

        # Save checkpoint
        torch.save(model_state, checkpoint_path)
        if is_best:
            torch.save(model_state, best_model_path)
            # Also copy scalers to best model directory if they exist
            if feature_scaler is not None or target_scaler is not None:
                best_scalers_path = os.path.join(self.local_checkpoint_dir, 'best_scalers.pt')
                shutil.copy2(scalers_path, best_scalers_path)
                self.logger.info(f"Saved best model checkpoint to {best_model_path} and scalers to {best_scalers_path}")
        else:
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Upload to S3
        try:
            s3_checkpoint_dir = f"{self.run_dir}/checkpoints"
            
            # Upload checkpoint
            s3_checkpoint_path = f"{s3_checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            self.fs.put(checkpoint_path, s3_checkpoint_path.replace('s3://', ''))
            
            # Upload scalers if they exist
            if feature_scaler is not None or target_scaler is not None:
                s3_scalers_path = f"{s3_checkpoint_dir}/scalers.pt"
                self.fs.put(scalers_path, s3_scalers_path.replace('s3://', ''))
                
                if is_best:
                    s3_best_scalers_path = f"{s3_checkpoint_dir}/best_scalers.pt"
                    self.fs.put(best_scalers_path, s3_best_scalers_path.replace('s3://', ''))
            
            if is_best:
                s3_best_model_path = f"{s3_checkpoint_dir}/best_model.pt"
                self.fs.put(best_model_path, s3_best_model_path.replace('s3://', ''))
            
            self.logger.info("Successfully uploaded checkpoint and scalers to S3")
            
        except Exception as e:
            self.logger.error(f"Failed to upload checkpoint to S3: {e}")
            raise
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/torch types to Python native types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        return obj

    def save_final_metrics(self, metrics):
        """Save final metrics with type conversion."""
        metrics = self._convert_to_serializable(metrics)
        path = f"{self.run_dir}/metrics/final_metrics.json"
        
        try:
            with self.fs.open(path.replace('s3://', ''), 'w') as f:
                json.dump(metrics, f, indent=2)
            self.logger.info("Saved final evaluation metrics")
        except Exception as e:
            self.logger.error(f"Failed to save final metrics: {e}")
            raise
    
    def log_metrics(self, metrics: dict, step: int, prefix: str = ''):
        """Log metrics with type conversion."""
        metrics = self._convert_to_serializable(metrics)
        
        # Add step number to metrics
        metrics['step'] = step
        
        # Create a more descriptive metrics filename
        metrics_file = f"{self.run_dir}/metrics/{prefix.lower()}_{step}_metrics.json"
        
        try:
            # Save as a new file for each step
            with self.fs.open(metrics_file.replace('s3://', ''), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Log metrics
            metrics_str = ' | '.join([f"{k}: {v:.6f}" if isinstance(v, float) 
                                    else f"{k}: {v}" for k, v in metrics.items()])
            self.logger.info(f"{prefix}Step {step}: {metrics_str}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            raise
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        best_checkpoint_path = os.path.join(self.local_checkpoint_dir, 'best_model.pt')
        if not os.path.exists(best_checkpoint_path):
            self.logger.warning("No best model checkpoint found.")
            return None

        try:
            # Add safe globals for loading
            torch.serialization.add_safe_globals({
                'StandardScaler': ('sklearn.preprocessing', 'StandardScaler'),
                'ModelEvaluator': ('evaluation.evaluator', 'ModelEvaluator'),
                'RunManager': ('evaluation.run_manager', 'RunManager')
            })
            
            # First try with weights_only=True
            try:
                checkpoint = torch.load(
                    best_checkpoint_path,
                    map_location=torch.device('cpu'),
                    weights_only=True
                )
            except Exception as e:
                self.logger.warning(f"Could not load with weights_only=True: {str(e)}")
                # Fallback to legacy loading method
                checkpoint = torch.load(
                    best_checkpoint_path,
                    map_location=torch.device('cpu')
                )
            
            self.logger.info(f"Loaded best model from {best_checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Error loading best model: {str(e)}")
            return None

    def save_xgboost_checkpoint(self, 
                          model,
                          val_metrics: Dict[str, float],
                          is_best: bool = False):
        """Save XGBoost model checkpoint locally first, then upload to S3."""
        # Save locally first
        local_filename = 'xgb_model_best.json' if is_best else 'xgb_model_latest.json'
        local_path = os.path.join(self.local_checkpoint_dir, local_filename)
        model.save_model(local_path)
        
        # Upload to S3
        s3_path = f"{self.run_dir}/checkpoints/{local_filename}"
        try:
            with self.fs.open(s3_path, 'wb') as f:
                with open(local_path, 'rb') as local_f:
                    f.write(local_f.read())
            self.logger.info(f"Saved XGBoost checkpoint to {s3_path}")
        except Exception as e:
            self.logger.error(f"Failed to upload XGBoost checkpoint to S3: {e}")
            raise

    def load_best_xgboost_model(self):
        """Load the best XGBoost model checkpoint."""
        best_checkpoint_path = os.path.join(self.local_checkpoint_dir, 'xgb_model_best.json')
        
        # Download from S3 if not exists locally
        if not os.path.exists(best_checkpoint_path):
            s3_path = f"{self.run_dir}/checkpoints/xgb_model_best.json"
            try:
                with self.fs.open(s3_path, 'rb') as f:
                    with open(best_checkpoint_path, 'wb') as local_f:
                        local_f.write(f.read())
            except Exception as e:
                self.logger.error(f"Failed to download XGBoost checkpoint from S3: {e}")
                return None
        
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor()
            model.load_model(best_checkpoint_path)
            self.logger.info(f"Loaded best XGBoost model from {best_checkpoint_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading best XGBoost model: {str(e)}")
            return None

    def save_log_file(self):
        """Save the current log file to S3."""
        try:
            # Use consistent path structure for logs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            s3_path = f"{self.run_dir}/logs/run_{timestamp}.log"
            
            # Create a temporary file with the log contents
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as temp_log:
                # Write the contents of our class-level buffer
                temp_log.write(RunManager.log_buffer.getvalue())
            
            # Upload to S3
            self.fs.put(temp_log.name, s3_path.replace('s3://', ''))
            self.logger.info(f"Saved log file to {s3_path}")
            
            # Cleanup
            os.unlink(temp_log.name)
            
        except Exception as e:
            self.logger.error(f"Failed to save log file: {e}")

    def cleanup_logging(self):
        """Close and remove all logging handlers."""
        try:
            # Only save logs if there are active handlers
            if self.logger.handlers:
                self.save_log_file()
            
            # Then cleanup handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
            
        except Exception as e:
            print(f"Error during logging cleanup: {e}")  # Use print as logger might be closed

    def __del__(self):
        """Cleanup temporary directory and logging handlers on deletion."""
        try:
            self.cleanup_logging()
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error during RunManager cleanup: {e}")
