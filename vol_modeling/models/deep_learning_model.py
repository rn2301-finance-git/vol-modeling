# models/deep_learning_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from evaluation.evaluator import ModelEvaluator
from evaluation.run_manager import RunManager
import logging
from models.loss_functions import CombinedVolatilityLoss


def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute the total gradient norm (L2) across all parameters.
    
    Args:
        model (nn.Module): PyTorch model with parameters.
        
    Returns:
        float: The L2 norm of all gradients.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


class MLPRegressorModel(nn.Module):
    """
    A fully-connected MLP for volatility forecasting, with optional
    BatchNorm, Dropout, and Kaiming initialization to match
    the improvements in the sequence model.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hd in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hd, bias=True))
            # BatchNorm
            layers.append(nn.BatchNorm1d(hd))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hd
        
        # Final layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional sanity check: if you want to raise an error on non-finite input
        if not torch.isfinite(x).all():
            raise ValueError(
                f"Non-finite input detected in MLPRegressorModel: min={x.min()}, max={x.max()}"
            )
        out = self.net(x)
        return out.squeeze(-1)


class VolatilityDataset(Dataset):
    """
    Simple Dataset to wrap a pandas DataFrame for model training,
    ensuring we can use the same DataLoader enhancements across models.
    """
    def __init__(self, df, feature_cols: List[str], target_col: str = "Y_log_vol_10min_lag_1m", 
                 target_scaler: Optional[StandardScaler] = None, fit_target: bool = True):
        super().__init__()
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        
        # Handle target scaling
        target_values = df[target_col].values.reshape(-1, 1)
        
        if target_scaler is None and fit_target:
            self.target_scaler = StandardScaler()
            target_scaled = self.target_scaler.fit_transform(target_values)
        elif target_scaler is not None:
            self.target_scaler = target_scaler
            target_scaled = target_scaler.transform(target_values)
        else:
            self.target_scaler = None
            target_scaled = target_values
            
        self.y = torch.tensor(target_scaled, dtype=torch.float32).squeeze()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
    
    def get_target_scaler(self) -> Optional[StandardScaler]:
        """Return the fitted target scaler if available."""
        return self.target_scaler


def validate_single_pass_mlp(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    evaluator: Optional[ModelEvaluator] = None
) -> Tuple[float, dict]:
    """
    Perform a single validation pass. Returns:
        val_loss (float): average validation loss
        val_metrics (dict): optional dictionary with additional metrics
    """
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            
            preds = model(x)
            loss = criterion(preds, y)
            total_val_loss += loss.item() * len(x)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    val_loss = total_val_loss / len(val_loader.dataset)
    
    val_metrics = {}
    if evaluator is not None:
        # Flatten
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_metrics = evaluator.evaluate_raw_arrays(all_preds, all_targets)
    
    return val_loss, val_metrics


def train_mlp_model(
    train_df,
    val_df,
    feature_cols: List[str],
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    hidden_dims: List[int] = [128, 64],
    dropout: float = 0.2,
    use_cuda: bool = False,
    run_manager: Optional[RunManager] = None,
    eval_train_set_frequency: int = 0,
    early_stopping_params: dict = None,
    gradient_accumulation_steps: int = 1,
    asymmetric_loss: bool = False
) -> Tuple[nn.Module, Optional[StandardScaler]]:
    """
    Train an MLP model with enhancements similar to `train_sequence_model`.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data frame with features + target
    val_df : pd.DataFrame
        Validation data frame with features + target
    feature_cols : List[str]
        List of feature columns
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Initial learning rate
    hidden_dims : List[int]
        List of hidden layer sizes
    dropout : float
        Dropout probability
    use_cuda : bool
        Whether to train on GPU
    run_manager : Optional[RunManager]
        Manager for logging and checkpointing
    eval_train_set_frequency : int
        How often (in epochs) to run a full evaluation pass on the training set
        (0 = never, 1 = every epoch, 5 = every 5 epochs, etc.)
    early_stopping_params : dict
        Dict with early stopping config:
            - patience: int
            - min_delta: float
            - mode: str
    gradient_accumulation_steps : int
        Number of batches to accumulate gradient before stepping optimizer
    asymmetric_loss : bool
        Whether to use asymmetric loss
    """
    if early_stopping_params is None:
        early_stopping_params = {
            "patience": 10,
            "min_delta": 1e-4,
            "mode": "min"
        }
    
    # Set up logging
    logger = run_manager.logger if run_manager else logging.getLogger("mlp_training")
    
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    scaler = GradScaler(enabled=use_cuda)  # For mixed precision training
    
    # Create Datasets and DataLoaders
    train_dataset = VolatilityDataset(train_df, feature_cols, target_col="Y_log_vol_10min_lag_1m")
    val_dataset = VolatilityDataset(val_df, feature_cols, target_col="Y_log_vol_10min_lag_1m", 
                                   target_scaler=train_dataset.get_target_scaler())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,            # HPC: more workers
        pin_memory=True,          # HPC
        persistent_workers=True,  # HPC
        prefetch_factor=4,        # HPC
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )
    
    # Log dataset info if run_manager is provided
    if run_manager:
        model_params = {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "device": str(device)
        }
        dataset_info = {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "feature_cols": feature_cols
        }
        run_manager.save_config(args={}, dataset_info=dataset_info, model_params=model_params)
    
    # Initialize model
    input_dim = len(feature_cols)
    model = MLPRegressorModel(input_dim, hidden_dims, dropout).to(device)
    
    # Initialize criterion based on asymmetric_loss parameter
    if asymmetric_loss:
        criterion = CombinedVolatilityLoss(
            alpha=0.7,  # Weight for asymmetric MSE
            over_penalty=2.0,  # Penalty for over-predictions
            under_penalty=1.0  # Penalty for under-predictions
        ).to(device)
        if run_manager:
            run_manager.logger.info("Using asymmetric loss function")
    else:
        criterion = nn.MSELoss()
        if run_manager:
            run_manager.logger.info("Using standard MSE loss function")
    
    # Criterion and optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,  # L2 regularization
        eps=1e-8
    )
    
    # LR scheduler (Reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    
    # Evaluator
    evaluator = ModelEvaluator(
        mode="full" if len(train_dataset) > 1000 else "subset",
        run_manager=run_manager
    )
    
    patience_counter = 0
    best_val_loss = float('inf')
    window_size = early_stopping_params["patience"]
    min_delta = early_stopping_params["min_delta"]
    mode = early_stopping_params["mode"]
    
    start_time = time.time()
    
    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # -----------------------------
            # 1) Training loop
            # -----------------------------
            model.train()
            train_loss = 0.0
            batch_start_time = time.time()
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                if not torch.isfinite(X_batch).all():
                    logger.error(f"Non-finite values in X_batch: min={X_batch.min()}, max={X_batch.max()}")
                    continue
                if not torch.isfinite(y_batch).all():
                    logger.error(f"Non-finite values in y_batch: min={y_batch.min()}, max={y_batch.max()}")
                    continue
                
                X_batch = X_batch.to(device, dtype=torch.float32, non_blocking=True)
                y_batch = y_batch.to(device, dtype=torch.float32, non_blocking=True)
                
                # Mixed-precision forward
                with autocast(enabled=use_cuda):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    # Scale down loss for gradient accumulation
                    loss = loss / max(1, gradient_accumulation_steps)
                
                # Check for non-finite loss
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss detected at batch {batch_idx}: {loss.item()}")
                    optimizer.zero_grad()
                    continue
                
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    # Unscale gradients first
                    scaler.unscale_(optimizer)
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * len(X_batch) * max(1, gradient_accumulation_steps)
                
                # Log batch metrics
                if run_manager and (batch_idx % 100 == 0):
                    batch_metrics = {
                        'batch_loss': float(loss.item()),
                        'learning_rate': float(optimizer.param_groups[0]['lr']),
                        'batch_time': time.time() - batch_start_time,
                        'gradient_norm': float(compute_grad_norm(model))
                    }
                    run_manager.log_metrics(batch_metrics, step=epoch * len(train_loader) + batch_idx, prefix='Batch')
                    
                    # Optional GPU memory log
                    if torch.cuda.is_available():
                        mem_alloc = torch.cuda.memory_allocated(device=device)
                        mem_reserved = torch.cuda.memory_reserved(device=device)
                        logger.info(
                            f"GPU Memory - Epoch {epoch+1}, Batch {batch_idx}: "
                            f"Allocated={mem_alloc/1e9:.2f}GB, Reserved={mem_reserved/1e9:.2f}GB"
                        )
                    batch_start_time = time.time()
            
            # Handle leftover accumulation
            if len(train_loader) % gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss /= len(train_loader.dataset)
            
            # -----------------------------
            # 2) Validation loop
            # -----------------------------
            val_loss, val_metrics = validate_single_pass_mlp(
                model=model,
                val_loader=val_loader,
                device=device,
                criterion=criterion,
                evaluator=evaluator
            )
            
            # Calculate train metrics if needed
            train_metrics_dict = {}
            if eval_train_set_frequency > 0 and epoch % eval_train_set_frequency == 0:
                # Evaluate on entire train set for metrics
                t_loss, t_metrics = validate_single_pass_mlp(
                    model=model,
                    val_loader=train_loader,
                    device=device,
                    criterion=criterion,
                    evaluator=evaluator
                )
                train_metrics_dict = t_metrics
            else:
                # Only store basic train loss
                train_metrics_dict['rmse'] = -1
                train_metrics_dict['rank_corr'] = 0.0
            
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s. "
                f"Val RMSE={val_metrics.get('rmse', float('inf')):.4f}, Val Loss={val_loss:.4f}"
            )
            
            # -----------------------------
            # 3) Logging / Early Stopping
            # -----------------------------
            if run_manager:
                metrics_to_log = {
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'val_rmse': float(val_metrics.get('rmse', -1.0)),
                    'val_rank_corr': float(val_metrics.get('rank_corr', 0.0)),
                    'train_rmse': float(train_metrics_dict.get('rmse', -1.0)),
                    'train_rank_corr': float(train_metrics_dict.get('rank_corr', 0.0)),
                    'learning_rate': float(optimizer.param_groups[0]['lr']),
                    'epoch_time': epoch_time
                }
                run_manager.log_metrics(metrics_to_log, step=epoch, prefix='Epoch')
            
            # Check improvement vs. best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                if run_manager:
                    run_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        val_loss=val_loss,
                        val_metrics=val_metrics,
                        is_best=True
                    )
            else:
                patience_counter += 1
            
            if patience_counter >= window_size:
                logger.info(
                    f"Early stopping triggered! No improvement for {patience_counter} epochs."
                )
                break
            
            # Periodic checkpoint (optional)
            if run_manager and epoch % 5 == 0:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    is_best=False
                )
                
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    except Exception as e:
        logger.error(f"Error during MLP training: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        total_training_time = time.time() - start_time
        logger.info(f"Total training time: {total_training_time:.2f} seconds")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # -----------------------------
    # Load best model before returning
    # -----------------------------
    if run_manager:
        best_ckpt = run_manager.load_best_model()
        if best_ckpt:
            model.load_state_dict(best_ckpt['model_state_dict'])
            logger.info(f"Loaded best model from epoch {best_ckpt['epoch']}")
    
    # In a typical pipeline, we might want to return a target_scaler
    # or None if you didn't scale the target. Adjust to your usage.
    # For completeness, we return None here (or a real scaler if you used one).
    
    return model, None


def evaluate_mlp_model(
    model: nn.Module,
    train_df,
    val_df,
    evaluator: ModelEvaluator,
    scaler: Optional[StandardScaler],
    feature_cols: List[str],
    notes: str = "",
    use_cuda: bool = False
):
    """
    Evaluate an MLP model on train and validation sets.
    Similar to evaluate_sequence_model for the sequence approach.
    """
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()
    
    # Evaluate on train
    train_preds, train_targets = get_predictions(model, train_df, feature_cols, scaler, device)
    train_metrics = evaluator.evaluate_predictions(
        (train_targets, train_preds)  # or pass as a DataFrame if you prefer
    )
    evaluator.log_results(
        model_name="mlp_model",
        metrics=train_metrics,
        params={"model_type": "MLP", "notes": notes},
        dataset_type="train"
    )
    
    # Evaluate on val
    val_preds, val_targets = get_predictions(model, val_df, feature_cols, scaler, device)
    val_metrics = evaluator.evaluate_predictions(
        (val_targets, val_preds)
    )
    evaluator.log_results(
        model_name="mlp_model",
        metrics=val_metrics,
        params={"model_type": "MLP", "notes": notes},
        dataset_type="validate"
    )
    
    # Optionally print the last few results
    evaluator.print_latest_summary(n_latest=2)


def get_predictions(
    model: nn.Module,
    df,
    feature_cols: List[str],
    scaler: Optional[StandardScaler],
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from the MLP model. Returns (predictions, targets).
    If you have a target scaler, invert or apply it as needed.
    """
    model.eval()
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["Y_log_vol_10min_lag_1m"].values.astype(np.float32)
    
    dataset = VolatilityDataset(df, feature_cols, "Y_log_vol_10min_lag_1m")
    loader = DataLoader(
        dataset,
        batch_size=2048,  # Could be bigger or smaller
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, dtype=torch.float32, non_blocking=True)
            preds = model(X_batch).cpu().numpy()
            # Unscale if you used a target scaler (example):
            # preds = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_preds, all_targets
