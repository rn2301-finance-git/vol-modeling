import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.loss_functions import SequenceVolatilityLoss, CombinedVolatilityLoss
from evaluation.evaluator import ModelEvaluator
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Any, Tuple
from evaluation.run_manager import RunManager
import os
import time
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import logging

def compute_grad_norm(model):
    """
    Compute the total gradient norm across all parameters.
    
    Args:
        model: PyTorch model with parameters
        
    Returns:
        float: The L2 norm of all gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

class CNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Add batch normalization after each conv layer and proper initialization
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite input detected: min={x.min()}, max={x.max()}")
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out.mean(dim=2)
        return out


class VolForecastNet(nn.Module):
    def __init__(self, short_in_dim, long_in_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        # hidden_dims is currently unused
        self.encoder_short = CNNEncoder(short_in_dim, hidden_dim)
        self.encoder_long = CNNEncoder(long_in_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights properly
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize final layer with smaller weights
        if isinstance(self.fc[-1], nn.Linear):
            nn.init.normal_(self.fc[-1].weight, mean=0.0, std=0.01)
            nn.init.constant_(self.fc[-1].bias, 0)

    def forward(self, x_short, x_long):
        # Ensure inputs are on same device as model
        if x_short.device != next(self.parameters()).device:
            x_short = x_short.to(next(self.parameters()).device)
        if x_long.device != next(self.parameters()).device:
            x_long = x_long.to(next(self.parameters()).device)
            
        # Input validation
        for x, name in [(x_short, 'short'), (x_long, 'long')]:
            if not torch.isfinite(x).all():
                raise ValueError(f"Non-finite input in {name} sequence: min={x.min()}, max={x.max()}")
        
        emb_s = self.encoder_short(x_short)
        emb_l = self.encoder_long(x_long)
        emb = torch.cat([emb_s, emb_l], dim=-1)
        out = self.fc(emb)
        return out.squeeze(-1)


def calculate_untransformed_rmse(model, loader, device):
    """Calculate RMSE in the original scale for sequence model."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_short, X_long, y in loader:
            X_short, X_long = X_short.to(device), X_long.to(device)
            preds = model(X_short, X_long).cpu().numpy()
            targets = y.numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    return np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))


def validate_single_pass(
    model, 
    val_loader, 
    device, 
    criterion, 
    evaluator=None
):
    """
    Perform a single validation pass:
      1) Accumulate the total val loss
      2) Collect model predictions and targets to compute metrics
    Returns:
      val_loss (float): average validation loss over dataset
      val_metrics (dict): optional, e.g. {"rmse": ..., "rank_corr": ...}
    """
    model.eval()
    
    total_val_loss = 0.0
    total_mse_loss = 0.0
    total_rank_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_short, X_long, y in val_loader:
            X_short = X_short.to(device, dtype=torch.float32, non_blocking=True)
            X_long  = X_long.to(device, dtype=torch.float32, non_blocking=True)
            y       = y.to(device, non_blocking=True)

            preds = model(X_short, X_long)

            # Accumulate loss
            loss = criterion(preds, y)
            if isinstance(criterion, CombinedVolatilityLoss):
                # Calculate individual loss components
                mse_loss = criterion.alpha * criterion.asym_mse(preds, y, 
                    over_penalty=criterion.over_penalty,
                    under_penalty=criterion.under_penalty)
                rank_loss = (1 - criterion.alpha) * criterion.rank_correlation_loss(preds, y)
                total_mse_loss += mse_loss.item() * len(X_short)
                total_rank_loss += rank_loss.item() * len(X_short)
            
            total_val_loss += loss.item() * len(X_short)

            # Store predictions & targets for later metric calculations
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Compute average loss
    val_loss = total_val_loss / len(val_loader.dataset)
    
    # Compute average component losses if using combined loss
    if isinstance(criterion, CombinedVolatilityLoss):
        avg_mse_loss = total_mse_loss / len(val_loader.dataset)
        avg_rank_loss = total_rank_loss / len(val_loader.dataset)
    else:
        avg_mse_loss = val_loss
        avg_rank_loss = 0.0

    # Optionally compute metrics in the same pass
    val_metrics = {
        'loss': val_loss,
        'mse_component': avg_mse_loss,
        'rank_component': avg_rank_loss
    }
    
    if evaluator is not None:
        # Flatten predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        # Evaluate metrics
        additional_metrics = evaluator.evaluate_raw_arrays(
            predictions=all_preds,
            targets=all_targets
        )
        val_metrics.update(additional_metrics)

    return val_loss, val_metrics


def train_sequence_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_cols,
    epochs: int = 10,
    learning_rate: float = 1e-5,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    use_cuda: bool = False,
    sequence_params: dict = None,
    run_manager: Optional[RunManager] = None,
    eval_train_set_frequency: int = 0,
    early_stopping_params: dict = None,
    gradient_accumulation_steps: int = 1,
    mode: str = "subset",
    target_scaler: Optional[StandardScaler] = None,
    asymmetric_loss: bool = False
):
    """
    Train sequence model with proper evaluation and logging.

    Parameters
    ----------
    ...
    eval_train_set_frequency : int
        How often to do a FULL pass on the train set to compute Spearman, etc.
        0 = never, 1 = every epoch, 5 = every 5 epochs, etc.
    early_stopping_params : dict, optional
        Dictionary containing early stopping configuration:
        - patience: int, maximum epochs to wait for improvement
        - min_delta: float, minimum relative improvement required
        - mode: str, mode for early stopping
    """
    if sequence_params is None:
        sequence_params = {
            'short_seq_len': 20,
            'long_seq_len': 60,
            'sample_every': 10,
            'batch_size': 64
        }
    
    if early_stopping_params is None:
        early_stopping_params = {
            "patience": 10,
            "min_delta": 1e-4,
            "mode": "min"
        }
    
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    scaler = GradScaler(enabled=use_cuda)  # Only enable for CUDA
    
    # Validate target scaler if provided
    if target_scaler is not None:
        logger = logging.getLogger('model_training')
        logger.info("\nTarget Scaler Statistics:")
        logger.info(f"Mean: {target_scaler.mean_[0]:.6f}")
        logger.info(f"Scale: {target_scaler.scale_[0]:.6f}")
        
        # Collect target statistics from training data
        all_targets = []
        for _, _, y in train_loader:
            all_targets.extend(y.numpy())
        all_targets = np.array(all_targets)
        
        logger.info("\nTraining Target Statistics:")
        logger.info(f"Mean: {np.mean(all_targets):.6f}")
        logger.info(f"Std: {np.std(all_targets):.6f}")
        logger.info(f"Min: {np.min(all_targets):.6f}")
        logger.info(f"Max: {np.max(all_targets):.6f}")
        
        # Verify scaler is properly fit
        if not hasattr(target_scaler, 'mean_') or not hasattr(target_scaler, 'scale_'):
            raise ValueError("Target scaler appears to not be properly fit")
        
        # Check for reasonable values
        if np.abs(target_scaler.mean_[0]) > 10 or target_scaler.scale_[0] <= 0:
            logger.warning("Target scaler parameters seem unusual - verify scaling is correct")
    
    # Log configuration if run_manager provided
    if run_manager:
        model_params = {
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'device': str(device),
            'loss_type': 'asymmetric' if asymmetric_loss else 'mse',
            **sequence_params,
            'eval_train_set_frequency': eval_train_set_frequency,
            'target_scaling': 'standard' if target_scaler is not None else 'none'
        }
        dataset_info = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'feature_count': len(feature_cols),
            'feature_names': feature_cols
        }
        run_manager.save_config(
            args={},  # Can be populated with command line args
            dataset_info=dataset_info,
            model_params=model_params
        )
    
    # Get input dimensions from first batch
    X_short, X_long, _ = next(iter(train_loader))
    short_in_dim = X_short.shape[-1]
    long_in_dim = X_long.shape[-1]
    
    # Initialize model and move to device with single call
    model = VolForecastNet(
        short_in_dim=short_in_dim,
        long_in_dim=long_in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    # Ensure criterion is on the same device
    if asymmetric_loss:
        criterion = CombinedVolatilityLoss(
            alpha=0.7,
            over_penalty=2.0,
            under_penalty=1.0
        ).to(device)
    else:
        criterion = nn.MSELoss().to(device)
    
    # Add gradient clipping configuration
    config = {
        "learning_rate": 1e-5,  # Keep the small learning rate
        "max_grad_norm": 1.0,   # Add explicit gradient clipping
        "warmup_steps": 100,    # Add warmup period
        # ... other configs ...
    }
    
    # Initialize optimizer with better defaults
    optimizer = torch.optim.AdamW(  # Switch to AdamW
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,  # Add L2 regularization
        eps=1e-8
    )
    
    # Use ReduceLROnPlateau instead of linear warmup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True if run_manager else False,
        min_lr=1e-6
    )
    
    # Create evaluator with correct mode
    evaluator = ModelEvaluator(
        mode=mode,
        over_penalty=2.0,
        under_penalty=1.0,
        run_manager=run_manager
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    val_loss_history = []  # NEW: Track validation loss history
    
    start_time = time.time()
    
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            if run_manager:
                run_manager.logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # -----------------------------
            # 1) Training loop
            # -----------------------------
            model.train()
            train_loss = 0.0
            train_metrics = {'rmse': None, 'rank_corr': None}
            batch_start_time = time.time()
            
            for batch_idx, (X_short, X_long, y) in enumerate(train_loader):
                # Add data validation
                for x, name in [(X_short, 'X_short'), (X_long, 'X_long'), (y, 'y')]:
                    if not torch.isfinite(x).all():
                        run_manager.logger.error(f"Non-finite values in {name}: min={x.min()}, max={x.max()}")
                        continue
                
                # Ensure minimum of 1 accumulation step
                gradient_accumulation_steps = max(gradient_accumulation_steps, 1)

                # Move data to device first
                X_short = X_short.to(device, dtype=torch.float32, non_blocking=True)
                X_long = X_long.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(enabled=use_cuda):
                    preds = model(X_short, X_long)
                    loss = criterion(preds, y)
                    # Proper gradient accumulation scaling
                    loss = loss / gradient_accumulation_steps
                
                # Skip bad loss values
                if not torch.isfinite(loss):
                    run_manager.logger.warning(f"Non-finite loss detected: {loss.item()}")
                    optimizer.zero_grad()
                    continue
                
                # Use scaler for mixed precision training
                scaler.scale(loss).backward()
                
                # Only step after accumulating gradients
                if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * len(X_short) * gradient_accumulation_steps
                
                # Add gradient norm to batch metrics logging
                if run_manager and (batch_idx % 100 == 0):
                    batch_metrics = {
                        'batch_loss': float(loss.item()),
                        'learning_rate': float(optimizer.param_groups[0]['lr']),
                        'batch_time': time.time() - batch_start_time,
                        'gradient_norm': float(compute_grad_norm(model))  # Add gradient norm to metrics
                    }
                    run_manager.log_metrics(
                        batch_metrics, 
                        step=epoch * len(train_loader) + batch_idx,
                        prefix='Batch'
                    )
                    
                    # (Optional) free up GPU memory occasionally
                    if torch.cuda.is_available() and batch_idx % 100 == 0:
                        memory_allocated = torch.cuda.memory_allocated(device=device)
                        memory_reserved = torch.cuda.memory_reserved(device=device)
                        run_manager.logger.info(
                            f"GPU Memory - Epoch {epoch+1}, Batch {batch_idx}: "
                            f"Allocated: {memory_allocated/1e9:.2f}GB, "
                            f"Reserved: {memory_reserved/1e9:.2f}GB"
                        )
                        torch.cuda.empty_cache()
                    
                    batch_start_time = time.time()
            
            # Handle any remaining gradients at end of epoch
            if (len(train_loader) % gradient_accumulation_steps) != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # -----------------------------
            # 2) Validation loop
            # -----------------------------
            val_loss, val_metrics = validate_single_pass(
                model=model,
                val_loader=val_loader,
                device=device,
                criterion=criterion,
                evaluator=evaluator
            )
            
            # Add validation loss to history
            val_loss_history.append(val_loss)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # -----------------------------------
            # 4) Logging / Early Stopping
            # -----------------------------------
            epoch_time = time.time() - epoch_start_time
            
            # Calculate training metrics if needed
            if eval_train_set_frequency > 0 and epoch % eval_train_set_frequency == 0:
                train_metrics = evaluator.evaluate_sequence_model(
                    model=model,
                    data_loader=train_loader,
                    device=device
                )
            
            # Log epoch metrics
            if run_manager:
                epoch_metrics = {
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'train_rmse': float(train_metrics.get('rmse', -1.0)) if train_metrics.get('rmse') is not None else -1.0,
                    'val_rmse': float(val_metrics.get('rmse', -1.0)) if val_metrics.get('rmse') is not None else -1.0,
                    'train_rank_corr': float(train_metrics.get('rank_corr', -1.0)) if train_metrics.get('rank_corr') is not None else -1.0,
                    'val_rank_corr': float(val_metrics.get('rank_corr', -1.0)) if val_metrics.get('rank_corr') is not None else -1.0,
                    'learning_rate': float(optimizer.param_groups[0]['lr']),
                    'epoch_time': epoch_time
                }
                run_manager.log_metrics(epoch_metrics, step=epoch, prefix='Epoch')
                run_manager.logger.info(
                    f"Epoch {epoch+1}/{epochs} done in {epoch_time:.2f}s. "
                    f"ValRMSE={val_metrics['rmse']:.4f}, ValLoss={val_loss:.4f}"
                )
            
            # NEW: Enhanced early stopping logic
            should_stop = False
            
            # 1. Check if current loss is best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
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
                
                # 2. Check for minimum improvement over window
                window_size = early_stopping_params["patience"]
                if len(val_loss_history) >= window_size:
                    window_start = val_loss_history[-window_size]
                    relative_improvement = (window_start - val_loss) / window_start
                    
                    if relative_improvement < early_stopping_params["min_delta"]:
                        patience_counter += 1
                        if run_manager:
                            run_manager.logger.info(
                                f"Insufficient improvement: {relative_improvement:.4f} < {early_stopping_params['min_delta']}"
                            )
            
            # Check if we should stop
            if patience_counter >= early_stopping_params["patience"]:
                if run_manager:
                    run_manager.logger.info(
                        f"Early stopping triggered! "
                        f"No improvement for {patience_counter} epochs"
                    )
                break
            
            # Optional periodic checkpoint
            if run_manager and epoch % 5 == 0:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    is_best=False
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log current learning rate
            if run_manager:
                current_lr = optimizer.param_groups[0]['lr']
                run_manager.logger.info(f"Current learning rate: {current_lr:.2e}")
            
            with torch.no_grad():
                grad_norm = compute_grad_norm(model)
                # Log gradients periodically and for very large values
                if batch_idx % 50 == 0:
                    run_manager.logger.info(f"Gradient norm at batch {batch_idx}: {grad_norm:.4f}")
                if grad_norm > 500.0:
                    run_manager.logger.info(
                        f"Very large gradient norm detected: {grad_norm:.4f} (epoch={epoch+1}, batch={batch_idx})"
                    )
            
    except KeyboardInterrupt:
        if run_manager:
            run_manager.logger.warning("Training interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    except Exception as e:
        if run_manager:
            run_manager.logger.error(f"Error during training: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        total_time = time.time() - start_time
        if run_manager:
            run_manager.logger.info(f"Total training time: {total_time:.2f} seconds")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # After training, load the best state if it exists
    if run_manager:
        best_state = run_manager.load_best_model()
        if best_state is not None:
            try:
                model.load_state_dict(best_state['model_state_dict'])
                run_manager.logger.info("Loaded best model state")
                
                final_metrics = {
                    'best_epoch': best_state['epoch'],
                    'best_val_loss': best_state['val_loss'],
                    'best_val_metrics': best_state['val_metrics'],
                    'total_epochs': epoch + 1,
                    'early_stopped': patience_counter >= early_stopping_params["patience"]
                }
            except Exception as e:
                run_manager.logger.error(f"Failed to load best model state: {e}")
                # Use current state for metrics if best state loading fails
                final_metrics = {
                    'best_epoch': epoch,
                    'best_val_loss': val_loss,
                    'best_val_metrics': val_metrics,
                    'total_epochs': epoch + 1,
                    'early_stopped': patience_counter >= early_stopping_params["patience"]
                }
        else:
            run_manager.logger.warning("No best model state found, using current model state")
            final_metrics = {
                'best_epoch': epoch,
                'best_val_loss': val_loss,
                'best_val_metrics': val_metrics,
                'total_epochs': epoch + 1,
                'early_stopped': patience_counter >= early_stopping_params["patience"]
            }
        
        run_manager.save_final_metrics(final_metrics)

    return model, target_scaler
