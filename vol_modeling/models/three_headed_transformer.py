"""
Three-headed transformer model for joint prediction of volatility and returns.

This module implements a transformer-based model with three prediction heads:
- Volatility prediction
- Returns prediction
- Confidence estimation

The model uses self-attention mechanisms to process sequential market data and
outputs both predictions and uncertainty estimates.

Key components:
- ThreeHeadedTransformer: Main model architecture
- train_three_headed_model: Training loop with logging
- evaluate_three_headed_model: Evaluation utilities
- predict_three_headed_model: Inference utilities
"""

import time
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from models.loss_functions import StabilizedCombinedLoss
from evaluation.run_manager import RunManager, setup_logging
import numpy as np
import math
from torch.utils.data import DataLoader
from evaluation.evaluator import ModelEvaluator
import logging

# Get logger for this module
logger = setup_logging().getChild('transformer')

class ThreeHeadedTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,  # Adjust based on your features
                 hidden_dim: int = 256,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 max_seq_length: int = 100,
                 gamma: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Add Layer Norm and Activation
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
        # Create positional encoding
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_seq_length, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Three-headed outputs with enhanced architectures
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.vol_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.ret_confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize the heads for stable initial predictions
        self._initialize_heads()
    
    def _initialize_heads(self):
        """
        Initialize the final layers of each head for stable initial predictions:
        - Returns and volatility heads: use default initialization
        - Logvar head: initialize to predict zero (variance=1)
        - Return confidence head: initialize to predict zero (sigmoid(0)=0.5)
        """
        # Get final layers
        vol_logvar_final = self.vol_logvar_head[-1]
        ret_conf_final = self.ret_confidence_head[-1]
        
        # Initialize logvar head to predict zero (variance=1)
        nn.init.zeros_(vol_logvar_final.weight)
        nn.init.zeros_(vol_logvar_final.bias)
        
        # Initialize return confidence head to predict zero (sigmoid(0)=0.5)
        nn.init.zeros_(ret_conf_final.weight)
        nn.init.zeros_(ret_conf_final.bias)
        
        # Returns and volatility heads keep default initialization
        # as they benefit from having non-zero initial predictions

    def forward(self, x: torch.Tensor) -> dict:
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = self.activation(x)  # Apply activation after embedding
        x = self.norm(x)  # Apply normalization
        
        x = x + self.position_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        
        # Use the last time step for prediction
        x_last = x[:, -1, :]
        return {
            'returns': self.return_head(x_last),
            'volatility': self.volatility_head(x_last),
            'vol_logvar': self.vol_logvar_head(x_last),
            'ret_confidence': self.ret_confidence_head(x_last)
        }



def train_three_headed_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    hidden_dim: int = 256,
    nhead: int = 8,
    num_layers: int = 3,
    dropout: float = 0.2,
    sequence_params: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1,
    use_cuda: bool = False,
    run_manager: Optional[RunManager] = None,
    eval_train_set_frequency: int = 5,
    asymmetric_loss: bool = False,
    early_stopping_params: Optional[Dict] = None,
    mode: str = "subset",
    feature_scaler: Optional[StandardScaler] = None,
    debug: bool = False,
    weight_decay: float = 0.01,
    use_parameter_groups: bool = False,
    warmup_steps: int = 1000,
    gamma: float = 0.1
) -> Tuple[ThreeHeadedTransformer, Tuple[StandardScaler, StandardScaler]]:
    """
    Train the three-headed transformer model with enhanced logging.
    
    Parameters:
      - train_loader, val_loader: DataLoaders yielding (inputs, targets) where targets is a dict with keys "returns" and "volatility".
      - input_dim: Should be set to the number of features per time step (typically len(selected_features)).
      - epochs: Number of training epochs.
      - learning_rate, hidden_dim, nhead, num_layers, dropout: Model hyperparameters.
      - sequence_params: Dictionary containing at least "sequence_length" (default 30).
      - gradient_accumulation_steps: For gradient accumulation.
      - use_cuda: Whether to use CUDA.
      - run_manager: Optional logging/checkpointing manager.
      - early_stopping_params: Dict with keys like "patience" and "min_delta".
      - mode: Data mode string (e.g. "subset").
      
    Returns:
      - A tuple of (trained model, (vol_scaler, ret_scaler)).
    """
    if sequence_params is None:
        sequence_params = {"sequence_length": 30, "sample_every": 10, "batch_size": 64}
    if early_stopping_params is None:
        early_stopping_params = {"patience": 10, "min_delta": 1e-4, "mode": "min"}
    
    assert feature_scaler is not None, "Feature scaler must be provided"
    # Adjust epochs if in debug mode
    if debug:
        original_epochs = epochs
        epochs = min(5, epochs)
        logger.info(f"Debug mode: Reducing epochs from {original_epochs} to {epochs}")
    
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # Simplified GradScaler initialization
    grad_scaler = GradScaler(enabled=use_cuda)
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    logger.info("Enabled PyTorch anomaly detection")
    
    # Initialize model with all required parameters
    model = ThreeHeadedTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_length=sequence_params.get("sequence_length", 30),
        gamma=gamma
    ).to(device)

    # Initialize optimizer based on parameter group flag
    if use_parameter_groups:
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
            # Add to no decay group if parameter is a bias or belongs to LayerNorm/BatchNorm
            if "bias" in name or "norm" in name or "position_encoding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Log parameter group sizes
        logger.info(f"Parameters with weight decay: {len(decay_params)}")
        logger.info(f"Parameters without weight decay: {len(no_decay_params)}")

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            eps=1e-8
        )
    else:
        # Use traditional single parameter group with uniform weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )

    # NEW: Set up warmup scheduler (updates every optimizer step)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps)
    )
    global_step = 0

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        min_lr=1e-6
    )

    # Initialize the combined loss function with gamma parameter
    loss_fn = StabilizedCombinedLoss(
        vol_weight=2.0,
        ret_weight=1.0,
        logvar_min=-3.0,
        logvar_max=3.0,
        gamma=gamma
    ).to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    val_loss_history = []
    total_training_start = time.time()

    # Track detailed statistics
    epoch_stats = {
        "vol_logvar_mean": [],
        "vol_logvar_std": [],
        "ret_conf_mean": [],
        "ret_conf_std": [],
        "vol_pred_mean": [],
        "vol_pred_std": [],
        "ret_pred_mean": [],
        "ret_pred_std": []
    }

    # Initialize scalers for targets
    vol_scaler = StandardScaler()
    ret_scaler = StandardScaler()
    
    # Collect all targets from training set to fit scalers
    all_vol_targets = []
    all_ret_targets = []
    
    logger.info("Fitting target scalers on training data...")
    with torch.no_grad():
        for inputs, targets in train_loader:
            all_vol_targets.append(targets['volatility'].numpy())
            all_ret_targets.append(targets['returns'].numpy())
    
    # Concatenate and fit scalers
    all_vol_targets = np.concatenate(all_vol_targets)
    all_ret_targets = np.concatenate(all_ret_targets)
    
    vol_scaler.fit(all_vol_targets.reshape(-1, 1))
    ret_scaler.fit(all_ret_targets.reshape(-1, 1))
    
    logger.info(f"Volatility scaler - mean: {vol_scaler.mean_[0]:.3f}, scale: {vol_scaler.scale_[0]:.3f}")
    logger.info(f"Returns scaler - mean: {ret_scaler.mean_[0]:.3f}, scale: {ret_scaler.scale_[0]:.3f}")
    
    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        batch_stats = {k: [] for k in epoch_stats.keys()}

        for batch_idx, batch in enumerate(train_loader):
            try:
                inputs, targets = batch
                inputs = inputs.to(device, non_blocking=True)
                targets_returns = torch.from_numpy(
                    ret_scaler.transform(targets['returns'].numpy().reshape(-1, 1))
                ).float().to(device)
                targets_vol = torch.from_numpy(
                    vol_scaler.transform(targets['volatility'].numpy().reshape(-1, 1))
                ).float().to(device)
                targets_dict = {'returns': targets_returns, 'volatility': targets_vol}

                # Forward pass
                outputs = model(inputs)

                # Log detailed statistics for this batch
                with torch.no_grad():
                    # Volatility logvar statistics
                    batch_stats["vol_logvar_mean"].append(outputs['vol_logvar'].mean().item())
                    batch_stats["vol_logvar_std"].append(outputs['vol_logvar'].std().item())
                    
                    # Return confidence statistics
                    ret_conf = torch.sigmoid(outputs['ret_confidence'])
                    batch_stats["ret_conf_mean"].append(ret_conf.mean().item())
                    batch_stats["ret_conf_std"].append(ret_conf.std().item())
                    
                    # Prediction statistics
                    batch_stats["vol_pred_mean"].append(outputs['volatility'].mean().item())
                    batch_stats["vol_pred_std"].append(outputs['volatility'].std().item())
                    batch_stats["ret_pred_mean"].append(outputs['returns'].mean().item())
                    batch_stats["ret_pred_std"].append(outputs['returns'].std().item())

                # Log every 100 batches
                if batch_idx % 100 == 0:
                    logger.info(
                        f"\nBatch {batch_idx} statistics:"
                        f"\nVolatility logvar - mean: {batch_stats['vol_logvar_mean'][-1]:.3f}, "
                        f"std: {batch_stats['vol_logvar_std'][-1]:.3f}"
                        f"\nReturn confidence - mean: {batch_stats['ret_conf_mean'][-1]:.3f}, "
                        f"std: {batch_stats['ret_conf_std'][-1]:.3f}"
                        f"\nPredictions:"
                        f"\n  Volatility - mean: {batch_stats['vol_pred_mean'][-1]:.3f}, "
                        f"std: {batch_stats['vol_pred_std'][-1]:.3f}"
                        f"\n  Returns - mean: {batch_stats['ret_pred_mean'][-1]:.3f}, "
                        f"std: {batch_stats['ret_pred_std'][-1]:.3f}"
                    )

                # Compute loss and handle NaN/Inf
                loss, loss_components = loss_fn(outputs, targets_dict)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, clamping to 0")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

                # Accumulate the original loss (before gradient accumulation scaling)
                train_loss_epoch += loss.item() * inputs.size(0)
                
                # Scale loss for gradient accumulation (only for backprop)
                loss = loss / gradient_accumulation_steps
                grad_scaler.scale(loss).backward()

                # Only update on accumulation steps or last batch
                if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    # Clip gradients
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                    # Update parameters
                    try:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer.zero_grad()
                        
                        # NEW: Update warmup scheduler if still in warmup phase
                        if global_step < warmup_steps:
                            warmup_scheduler.step()
                        global_step += 1

                    except RuntimeError as e:
                        logger.error(f"Error during optimizer step: {str(e)}")
                        optimizer.zero_grad()
                        continue

            except RuntimeError as e:
                logger.error(f"Runtime error in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()
                continue

        # Normalize by dataset size at the end of epoch
        train_loss_epoch /= len(train_loader.dataset)

        # Compute epoch-level statistics
        for key in epoch_stats:
            if batch_stats[key]:  # Check if we have any values
                epoch_stats[key].append(np.mean(batch_stats[key]))

        # Log epoch-level statistics
        if run_manager:
            run_manager.log_metrics({
                "vol_logvar_mean": epoch_stats["vol_logvar_mean"][-1],
                "vol_logvar_std": epoch_stats["vol_logvar_std"][-1],
                "ret_conf_mean": epoch_stats["ret_conf_mean"][-1],
                "ret_conf_std": epoch_stats["ret_conf_std"][-1],
                "vol_pred_mean": epoch_stats["vol_pred_mean"][-1],
                "vol_pred_std": epoch_stats["vol_pred_std"][-1],
                "ret_pred_mean": epoch_stats["ret_pred_mean"][-1],
                "ret_pred_std": epoch_stats["ret_pred_std"][-1]
            }, step=epoch, prefix='epoch_stats')

        # Print epoch summary
        logger.info(
            f"\nEpoch {epoch} Summary:"
            f"\nVolatility logvar - mean: {epoch_stats['vol_logvar_mean'][-1]:.3f}, "
            f"std: {epoch_stats['vol_logvar_std'][-1]:.3f}"
            f"\nReturn confidence - mean: {epoch_stats['ret_conf_mean'][-1]:.3f}, "
            f"std: {epoch_stats['ret_conf_std'][-1]:.3f}"
            f"\nPredictions:"
            f"\n  Volatility - mean: {epoch_stats['vol_pred_mean'][-1]:.3f}, "
            f"std: {epoch_stats['vol_pred_std'][-1]:.3f}"
            f"\n  Returns - mean: {epoch_stats['ret_pred_mean'][-1]:.3f}, "
            f"std: {epoch_stats['ret_pred_std'][-1]:.3f}"
        )

        # Run validation.
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets_returns = torch.from_numpy(
                    ret_scaler.transform(targets['returns'].numpy().reshape(-1, 1))
                ).float().to(device)
                targets_vol = torch.from_numpy(
                    vol_scaler.transform(targets['volatility'].numpy().reshape(-1, 1))
                ).float().to(device)
                targets_dict = {'returns': targets_returns, 'volatility': targets_vol}

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    outputs = model(inputs)
                    loss, _ = loss_fn(outputs, targets_dict)
                val_loss_epoch += loss.item() * inputs.size(0)  # Multiply by batch size

        val_loss_epoch /= len(val_loader.dataset)
        scheduler.step(val_loss_epoch)
        val_loss_history.append(val_loss_epoch)
        epoch_time = time.time() - total_training_start

        # Log metrics.
        if run_manager:
            run_manager.log_metrics({
                'train_loss': train_loss_epoch,
                'val_loss': val_loss_epoch,
                'epoch_time': epoch_time
            }, step=epoch, prefix='training')

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f} (Time: {epoch_time:.2f}s)")

        # Early stopping logic.
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            if run_manager:
                run_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss_epoch,
                    val_metrics={'total_loss': val_loss_epoch},
                    is_best=True,
                    feature_scaler=feature_scaler,
                    target_scaler=(vol_scaler, ret_scaler)
                )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_params["patience"]:
                print(f"Early stopping triggered after epoch {epoch+1} with no improvement.")
                break

    total_training_time = time.time() - total_training_start
    print(f"Total training time: {total_training_time:.2f} seconds")

    if run_manager:
        best_state = run_manager.load_best_model()
        if best_state is not None:
            model.load_state_dict(best_state['model_state_dict'])
            print("Loaded best model state from checkpoint.")
    
    # Log final training dynamics using log_metrics instead of log_figure
    if run_manager:
        for key in epoch_stats:
            if epoch_stats[key]:  # Check if we have any values
                metrics_dict = {
                    f"epoch_{epoch}": value 
                    for epoch, value in enumerate(epoch_stats[key])
                }
                run_manager.log_metrics(metrics_dict, step=epochs-1, prefix=f'training_dynamics/{key}')
                
                run_manager.log_metrics({
                    'mean': np.mean(epoch_stats[key]),
                    'std': np.std(epoch_stats[key]),
                    'min': np.min(epoch_stats[key]),
                    'max': np.max(epoch_stats[key])
                }, step=epochs-1, prefix=f'final_stats/{key}')

        run_manager.log_metrics({
            "total_epochs": epochs,
            "final_train_loss": train_loss_epoch,
            "best_val_loss": best_val_loss
        }, step=epochs-1, prefix='training_summary')

    # Return the trained model and scalers
    return model, (vol_scaler, ret_scaler)



def evaluate_three_headed_model(
    model: nn.Module,
    train_loader,
    val_loader,
    evaluator : ModelEvaluator,
    run_manager: Optional[RunManager] = None,
    notes: str = "",
    use_cuda: bool = False
) -> dict:
    """
    Evaluate the three-headed transformer model on both training and validation sets.
    
    Parameters:
    -----------
    model: The trained three-headed transformer model.
    train_loader: DataLoader for training data.
    val_loader: DataLoader for validation data.
    evaluator: A ModelEvaluator instance for computing and logging metrics.
    run_manager: Optional RunManager for logging.
    notes: Additional notes for logging.
    use_cuda: Whether to use GPU acceleration.
      
    Returns:
    --------
    dict: Dictionary containing evaluation metrics
    """
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    loss_fn = StabilizedCombinedLoss(vol_weight=1.0, ret_weight=1.0).to(device)
    
    def validate_pass(data_loader):
        model.eval()
        total_loss = 0.0
        
        all_preds_returns = []
        all_targets_returns = []
        all_preds_vol = []
        all_targets_vol = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                targets_returns = targets['returns'].to(device, dtype=torch.float32, non_blocking=True)
                targets_vol = targets['volatility'].to(device, dtype=torch.float32, non_blocking=True)
                targets_dict = {'returns': targets_returns, 'volatility': targets_vol}
                
                with autocast(enabled=use_cuda):
                    outputs = model(inputs)
                    loss, _ = loss_fn(outputs, targets_dict)
                
                total_loss += loss.item() * inputs.size(0)
                
                all_preds_returns.append(outputs['returns'].cpu().numpy())
                all_targets_returns.append(targets_returns.cpu().numpy())
                all_preds_vol.append(outputs['volatility'].cpu().numpy())
                all_targets_vol.append(targets_vol.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader.dataset)
        
        preds_returns = np.concatenate(all_preds_returns, axis=0)
        targets_returns = np.concatenate(all_targets_returns, axis=0)
        preds_vol = np.concatenate(all_preds_vol, axis=0)
        targets_vol = np.concatenate(all_targets_vol, axis=0)
        
        metrics_returns = evaluator.evaluate_raw_arrays(preds_returns, targets_returns)
        metrics_vol = evaluator.evaluate_raw_arrays(preds_vol, targets_vol)
        
        return {
            'loss': avg_loss,
            'returns_rmse': metrics_returns['rmse'],
            'returns_rank_corr': metrics_returns['rank_corr'],
            'volatility_rmse': metrics_vol['rmse'],
            'volatility_rank_corr': metrics_vol['rank_corr'],
            'rmse': metrics_vol['rmse'],  # Add standard metrics keys
            'rank_corr': metrics_vol['rank_corr']  # Add standard metrics keys
        }
    
    # Evaluate on train and validation sets
    train_metrics = validate_pass(train_loader)
    val_metrics = validate_pass(val_loader)
    
    # Add confidence calibration evaluation
    train_calibration = evaluate_confidence_calibration(model, train_loader, device)
    val_calibration = evaluate_confidence_calibration(model, val_loader, device)
    
    # Combine all metrics
    metrics = {
        'train_loss': train_metrics['loss'],
        'val_loss': val_metrics['loss'],
        'train_returns_rmse': train_metrics['returns_rmse'],
        'train_returns_rank_corr': train_metrics['returns_rank_corr'],
        'train_volatility_rmse': train_metrics['volatility_rmse'],
        'train_volatility_rank_corr': train_metrics['volatility_rank_corr'],
        'val_returns_rmse': val_metrics['returns_rmse'],
        'val_returns_rank_corr': val_metrics['returns_rank_corr'],
        'val_volatility_rmse': val_metrics['volatility_rmse'],
        'val_volatility_rank_corr': val_metrics['volatility_rank_corr'],
        'rmse': val_metrics['rmse'],  # Standard metrics for compatibility
        'rank_corr': val_metrics['rank_corr'],  # Standard metrics for compatibility
        'train_vol_calibration_error': train_calibration['vol_confidence_calibration']['calibration_error'],
        'train_ret_calibration_error': train_calibration['ret_confidence_calibration']['calibration_error'],
        'val_vol_calibration_error': val_calibration['vol_confidence_calibration']['calibration_error'],
        'val_ret_calibration_error': val_calibration['ret_confidence_calibration']['calibration_error']
    }
    
    # Log results with evaluator
    evaluator.log_results(
        model_name="three_headed_transformer",
        metrics=metrics,
        params={"model_type": "ThreeHeaded", "notes": notes},
        dataset_type="validate"
    )
    
    # Log calibration bins if run_manager is available
    if run_manager:
        run_manager.log_metrics({
            'vol_bins': train_calibration['vol_confidence_calibration']['bins'],
            'ret_bins': train_calibration['ret_confidence_calibration']['bins']
        }, step=0, prefix='train_calibration')
        
        run_manager.log_metrics({
            'vol_bins': val_calibration['vol_confidence_calibration']['bins'],
            'ret_bins': val_calibration['ret_confidence_calibration']['bins']
        }, step=0, prefix='val_calibration')
    
    return metrics  # Return the metrics dictionary


def compute_calibration_metrics(conf_error_pairs):
    """
    Compute calibration metrics for a list of (confidence, error) pairs.
    
    Parameters:
    -----------
    conf_error_pairs : List[Tuple[np.ndarray, np.ndarray]]
        List of (confidence, error) pairs.
        
    Returns:
    --------
    Dict containing calibration error and bin statistics.
    
    Note:
    -----
    The "true confidence" is computed as 1 / (1 + average_error) for each bin.
    The calibration error is a weighted average of the absolute differences between
    the predicted confidence and this true confidence.
    """
    # If the input list is empty, return zeros.
    if not conf_error_pairs:
        return {'calibration_error': 0.0, 'bins': []}
    
    # Flatten the arrays if they're batched
    try:
        confidences = np.concatenate([conf.flatten() for conf, _ in conf_error_pairs])
        errors = np.concatenate([err.flatten() for _, err in conf_error_pairs])
    except Exception as e:
        logger = logging.getLogger('transformer')
        logger.error(f"Error concatenating arrays: {str(e)}")
        logger.error(f"Shapes: {[conf.shape for conf, _ in conf_error_pairs]}, {[err.shape for _, err in conf_error_pairs]}")
        raise
    
    # Sort by confidence
    sort_idx = np.argsort(confidences)
    confidences = confidences[sort_idx]
    errors = errors[sort_idx]
    
    total_examples = len(confidences)
    # Avoid zero bin size if total_examples < 10
    bin_size = total_examples // 10 if total_examples >= 10 else 1
    
    bins = []
    for i in range(0, total_examples, bin_size):
        end_idx = min(i + bin_size, total_examples)
        bin_conf = confidences[i:end_idx]
        bin_err = errors[i:end_idx]
        
        if len(bin_conf) > 0:  # Only process non-empty bins
            avg_confidence = np.mean(bin_conf)
            avg_error = np.mean(bin_err)
            bins.append({
                'avg_confidence': float(avg_confidence),  # Convert to Python float
                'avg_error': float(avg_error),            # Convert to Python float
                'size': len(bin_conf)
            })
    
    # Compute calibration error: weighted average absolute difference
    calibration_error = sum(
        abs(bin['avg_confidence'] - (1 / (1 + bin['avg_error']))) * bin['size']
        for bin in bins
    ) / total_examples
    
    return {
        'calibration_error': float(calibration_error),
        'bins': bins
    }

def evaluate_confidence_calibration(model, data_loader, device):
    """
    Evaluate how well calibrated the model's confidence predictions are.
    """
    model.eval()
    vol_conf_pairs = []
    ret_conf_pairs = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets_vol = targets['volatility'].to(device)
            targets_ret = targets['returns'].to(device)
            
            outputs = model(inputs)
            
            # Get confidence scores
            vol_conf = torch.exp(-outputs['vol_logvar']).cpu().numpy()
            ret_conf = torch.sigmoid(outputs['ret_confidence']).cpu().numpy()
            
            # Get prediction errors
            vol_error = torch.abs(outputs['volatility'] - targets_vol).cpu().numpy()
            ret_error = torch.abs(outputs['returns'] - targets_ret).cpu().numpy()
            
            # Store pairs for each batch
            vol_conf_pairs.append((vol_conf, vol_error))
            ret_conf_pairs.append((ret_conf, ret_error))
    
    return {
        'vol_confidence_calibration': compute_calibration_metrics(vol_conf_pairs),
        'ret_confidence_calibration': compute_calibration_metrics(ret_conf_pairs)
    }

def predict_three_headed_model(
    model: nn.Module,
    dataloader: DataLoader,
    target_scaler: Optional[Any] = None,
    use_cuda: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate predictions using the three-headed transformer model.
    
    Parameters
    ----------
    model : nn.Module
        Trained transformer model
    dataloader : DataLoader
        DataLoader containing sequences to predict
    target_scaler : Optional[Any]
        Either a single scaler or a tuple of (vol_scaler, ret_scaler)
    use_cuda : bool
        Whether to use GPU if available
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing predictions for volatility, returns, and confidence scores
    """
    logger = logging.getLogger('transformer')
    logger.info("Generating predictions with three-headed transformer...")
    
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_vol_preds = []
    all_ret_preds = []
    all_vol_conf = []
    all_ret_conf = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            # Move inputs to device
            inputs = inputs.to(device)
            
            # Get model outputs
            outputs = model(inputs)
            
            # Extract predictions and convert to numpy
            vol_preds = outputs['volatility'].cpu().numpy()
            ret_preds = outputs['returns'].cpu().numpy()
            
            # Convert confidence scores
            vol_conf = torch.exp(-outputs['vol_logvar']).cpu().numpy()  # Transform logvar to confidence
            ret_conf = torch.sigmoid(outputs['ret_confidence']).cpu().numpy()  # Already in [0,1]
            
            # Collect batch predictions
            all_vol_preds.append(vol_preds)
            all_ret_preds.append(ret_preds)
            all_vol_conf.append(vol_conf)
            all_ret_conf.append(ret_conf)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx+1} batches...")
    
    # Concatenate all predictions and ensure they are 1D arrays
    vol_predictions = np.concatenate(all_vol_preds).ravel()
    ret_predictions = np.concatenate(all_ret_preds).ravel()
    vol_confidence = np.concatenate(all_vol_conf).ravel()
    ret_confidence = np.concatenate(all_ret_conf).ravel()
    
    # Apply inverse transform if scaler provided
    if target_scaler is not None:
        if isinstance(target_scaler, tuple):
            # Unpack the tuple of scalers
            vol_scaler, ret_scaler = target_scaler
            logger.info("Using separate scalers for volatility and returns")
            vol_predictions = vol_scaler.inverse_transform(vol_predictions.reshape(-1, 1)).squeeze()
            ret_predictions = ret_scaler.inverse_transform(ret_predictions.reshape(-1, 1)).squeeze()
        else:
            # Use single scaler for both (legacy behavior)
            logger.warning("Using single scaler for both volatility and returns")
            vol_predictions = target_scaler.inverse_transform(vol_predictions.reshape(-1, 1)).squeeze()
            ret_predictions = target_scaler.inverse_transform(ret_predictions.reshape(-1, 1)).squeeze()
    
    logger.info("Prediction generation completed.")
    logger.info(f"Generated {len(vol_predictions):,} predictions")
    
    return {
        'volatility': vol_predictions,
        'returns': ret_predictions,
        'vol_confidence': vol_confidence,
        'ret_confidence': ret_confidence
    }