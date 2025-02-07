import torch
import torch.nn as nn
import math

def asym_mse(pred, target, over_penalty=2.0, under_penalty=1.0):
    """
    Asymmetric MSE loss that penalizes over-predictions more heavily.
    
    Parameters:
    -----------
    pred : torch.Tensor
        Model predictions
    target : torch.Tensor
        True values
    over_penalty : float
        Multiplier for loss when prediction > target
    under_penalty : float
        Multiplier for loss when prediction < target
    
    Returns:
    --------
    torch.Tensor
        Computed loss value
    """
    diff = pred - target
    over_mask = (diff > 0).float()
    under_mask = 1 - over_mask
    return torch.mean(over_penalty * over_mask * diff**2 +
                     under_penalty * under_mask * diff**2)

def rank_correlation_loss(pred, target):
    """
    Compute loss based on rank correlation.
    
    Parameters:
    -----------
    pred : torch.Tensor
        Model predictions
    target : torch.Tensor
        True values
    
    Returns:
    --------
    torch.Tensor
        MSE between rank positions
    """
    pred_rank = torch.argsort(torch.argsort(pred))
    target_rank = torch.argsort(torch.argsort(target))
    return torch.mean((pred_rank.float() - target_rank.float()) ** 2)

class CombinedVolatilityLoss(nn.Module):
    """
    Combines asymmetric MSE and rank correlation losses.
    """
    def __init__(self, alpha=0.7, over_penalty=2.0, under_penalty=1.0):
        """
        Parameters:
        -----------
        alpha : float
            Weight for asymmetric MSE (1-alpha for rank loss)
        over_penalty : float
            Penalty multiplier for over-predictions
        under_penalty : float
            Penalty multiplier for under-predictions
        """
        super().__init__()
        self.alpha = alpha
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty
    
    def forward(self, pred, target):
        """
        Compute combined loss.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Model predictions
        target : torch.Tensor
            True values
        
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        mse_loss = asym_mse(
            pred, target, 
            over_penalty=self.over_penalty,
            under_penalty=self.under_penalty
        )
        rank_loss = rank_correlation_loss(pred, target)
        
        return self.alpha * mse_loss + (1 - self.alpha) * rank_loss

class SequenceVolatilityLoss(CombinedVolatilityLoss):
    """
    Extension of CombinedVolatilityLoss for sequence models.
    Can be extended with sequence-specific loss components if needed.
    """
    def __init__(self, alpha=0.7, over_penalty=2.0, under_penalty=1.0):
        super().__init__(alpha, over_penalty, under_penalty)


class ConfidenceVolatilityLoss(nn.Module):
    """
    Implements heteroskedastic Gaussian NLL for volatility prediction:
    NLL = 0.5 * log(var) + 0.5 * ((y - mu)^2 / var)
    """
    def __init__(self):
        super().__init__()
        self.constant = 0.5 * math.log(2 * math.pi)


    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: predicted volatility [batch, 1]
            logvar: log variance (confidence) [batch, 1]
            target: actual volatility [batch, ...]
        """
        var = torch.exp(logvar)
        nll = 0.5 * logvar + 0.5 * ((target - mu)**2 / var) + self.constant
        return nll.mean()

class ConfidenceReturnLoss(nn.Module):
    """
    Implements a confidence-weighted MSE loss for returns with an entropy penalty:
    L = confidence * MSE + alpha * log(confidence)

    The entropy penalty prevents the model from always predicting maximum confidence.
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, confidence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted return [batch, 1]
            confidence: confidence score [batch, 1]
            target: actual return [batch, 1]
        """
        # Normalize confidence to [0,1] using sigmoid
        confidence = torch.sigmoid(confidence)

        # Compute MSE
        mse = (pred - target)**2

        # Weight MSE by confidence and add entropy penalty
        loss = (confidence * mse).mean() - self.alpha * torch.log(confidence + 1e-8).mean()

        return loss

class CombinedConfidenceLoss(nn.Module):
    """
    Combines volatility and return confidence losses with optional weighting
    """
    def __init__(self, vol_weight: float = 1.0, ret_weight: float = 1.0):
        super().__init__()
        self.vol_loss = ConfidenceVolatilityLoss()
        self.ret_loss = ConfidenceReturnLoss()
        self.vol_weight = vol_weight
        self.ret_weight = ret_weight

    def forward(self, outputs: dict, targets: dict) -> tuple:
        """
        Args:
            outputs: dict containing:
                - 'volatility': predicted volatility
                - 'vol_logvar': log variance for volatility
                - 'returns': predicted returns
                - 'ret_confidence': return confidence
            targets: dict containing:
                - 'volatility': actual volatility
                - 'returns': actual returns

        Returns:
            tuple of (total_loss, dict of individual losses)
        """
        # Compute individual losses
        vol_loss = self.vol_loss(
            outputs['volatility'],
            outputs['vol_logvar'],
            targets['volatility']
        )

        ret_loss = self.ret_loss(
            outputs['returns'],
            outputs['ret_confidence'],
            targets['returns']
        )

        # Combine losses
        total_loss = self.vol_weight * vol_loss + self.ret_weight * ret_loss

        return total_loss, {
            'vol_loss': vol_loss.item(),
            'ret_loss': ret_loss.item(),
            'total_loss': total_loss.item()
        }

class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced combined loss function that implements:
    1. Heteroscedastic loss for volatility prediction with regularization
    2. Confidence-weighted loss for returns prediction
    """
    def __init__(self, vol_weight: float = 1.0, ret_weight: float = 1.0, 
                 confidence_reg: float = 0.1, logvar_reg: float = 0.1):
        """
        Parameters:
        -----------
        vol_weight : float
            Weight for the volatility loss component
        ret_weight : float
            Weight for the returns loss component
        confidence_reg : float
            Regularization coefficient for confidence entropy
        logvar_reg : float
            Regularization coefficient for logvar prior
        """
        super().__init__()
        self.vol_weight = vol_weight
        self.ret_weight = ret_weight
        self.confidence_reg = confidence_reg
        self.logvar_reg = logvar_reg
        
    def forward(self, outputs: dict, targets: dict) -> tuple:
        """
        Compute combined loss.
        """
        # Volatility loss with heteroscedastic component
        clamped_logvar = torch.clamp(outputs['vol_logvar'], min=-3.0, max=3.0)
        vol_loss = 0.5 * torch.exp(-clamped_logvar) * (outputs['volatility'] - targets['volatility']).pow(2) + \
                  0.5 * clamped_logvar + \
                  self.logvar_reg * clamped_logvar.pow(2)  # Add regularization term

        # Returns loss with confidence weighting
        ret_conf = torch.sigmoid(outputs['ret_confidence'])
        ret_loss = ret_conf * (outputs['returns'] - targets['returns']).pow(2) \
                  - self.confidence_reg * torch.log(ret_conf + 1e-8)
                  
        # Combine losses with weights
        total_loss = self.vol_weight * vol_loss.mean() + \
                    self.ret_weight * ret_loss.mean()
                    
        return total_loss, {
            'vol_loss': vol_loss.mean().item(),
            'ret_loss': ret_loss.mean().item(),
            'total_loss': total_loss.item()
        }



class StabilizedConfidenceVolatilityLoss(nn.Module):
    """
    Implements numerically stabilized heteroskedastic Gaussian NLL for volatility prediction
    with regularization to prevent overconfident predictions
    """
    def __init__(self, logvar_min=-3.0, logvar_max=3.0, eps=1e-6, reg_lambda=0.1):
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.eps = eps
        self.reg_lambda = reg_lambda  # Weight for regularization term
        self.constant = 0.5 * math.log(2 * math.pi)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: predicted volatility [batch, 1]
            logvar: log variance (confidence) [batch, 1] 
            target: actual volatility [batch, ...]
        """
        # Replace any NaN/Inf with safe values
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=self.logvar_max, neginf=self.logvar_min)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        # Compute loss terms carefully
        sq_error = (target - mu).pow(2)
        weighted_sq_error = 0.5 * torch.exp(-logvar) * sq_error
        log_var_term = 0.5 * logvar
        
        # Add regularization term to penalize extreme logvar values
        # This is effectively a Gaussian prior on logvar centered at 0
        reg_term = self.reg_lambda * logvar.pow(2)
        
        # Combine terms and add constant
        nll = log_var_term + weighted_sq_error + self.constant + reg_term
        
        return nll.mean()

class StabilizedConfidenceReturnLoss(nn.Module):
    """
    Implements a numerically stable confidence-weighted MSE loss for returns
    """
    def __init__(self, alpha=0.1, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, confidence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted return [batch, 1]
            confidence: confidence score [batch, 1]
            target: actual return [batch, 1]
        """
        # Replace any NaN/Inf with safe values
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        confidence = torch.nan_to_num(confidence, nan=0.5)  # Default to medium confidence
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale confidence to [0,1] with careful normalization
        confidence = torch.sigmoid(confidence)
        confidence = torch.clamp(confidence, min=self.eps, max=1-self.eps)
        
        # Compute MSE with clamping
        mse = torch.clamp((pred - target).pow(2), max=100.0)
        
        # Weight MSE by confidence and add entropy regularizer
        weighted_mse = (confidence * mse).mean()
        entropy_reg = -self.alpha * torch.log(confidence).mean()
        
        return weighted_mse + entropy_reg

class StabilizedCombinedLoss(nn.Module):
    """
    Combines stabilized volatility and return confidence losses with robust weighting
    """
    def __init__(self, vol_weight=1.0, ret_weight=1.0, logvar_min=-5.0, logvar_max=5.0, gamma=0.1):
        super().__init__()
        # Initialize with gamma parameter
        self.vol_loss = CalibratedConfidenceVolatilityLoss(
            logvar_min=logvar_min, 
            logvar_max=logvar_max, 
            eps=1e-6, 
            reg_lambda=0.01,
            gamma=gamma  # Pass gamma to volatility loss
        )
        self.ret_loss = StabilizedConfidenceReturnLoss(alpha=0.02)
        self.vol_weight = vol_weight
        self.ret_weight = ret_weight

    def check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Helper to check tensor validity and print diagnostics"""
        if tensor is None:
            print(f"Warning: {name} is None")
            return False
            
        if not isinstance(tensor, torch.Tensor):
            print(f"Warning: {name} is not a tensor, got {type(tensor)}")
            return False
            
        if tensor.nelement() == 0:
            print(f"Warning: {name} is empty")
            return False
            
        n_nan = torch.isnan(tensor).sum().item()
        n_inf = torch.isinf(tensor).sum().item()
        
        if n_nan > 0 or n_inf > 0:
            print(f"Warning: {name} contains {n_nan} NaNs and {n_inf} Infs")
            print(f"{name} stats - min: {tensor[torch.isfinite(tensor)].min().item():.3f}, "
                  f"max: {tensor[torch.isfinite(tensor)].max().item():.3f}")
            
        return True

    def forward(self, outputs: dict, targets: dict) -> tuple:
        """
        Args:
            outputs: dict containing model outputs
            targets: dict containing true values
        Returns:
            tuple of (total_loss, dict of individual losses)
        """
        # Validate inputs exist
        required_outputs = ['volatility', 'vol_logvar', 'returns', 'ret_confidence']
        required_targets = ['volatility', 'returns']
        
        for key in required_outputs:
            if key not in outputs:
                raise KeyError(f"Missing required output: {key}")
        for key in required_targets:
            if key not in targets:
                raise KeyError(f"Missing required target: {key}")
        
        # Print shapes and basic stats for debugging
        print("\nInput tensor shapes and stats:")
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape={v.shape}, min={v.min().item():.3f}, max={v.max().item():.3f}")
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                print(f"target_{k}: shape={v.shape}, min={v.min().item():.3f}, max={v.max().item():.3f}")
        
        # Compute losses (will handle NaN/Inf internally)
        vol_loss = self.vol_loss(
            outputs['volatility'],
            outputs['vol_logvar'],
            targets['volatility']
        )
        
        ret_loss = self.ret_loss(
            outputs['returns'],
            outputs['ret_confidence'],
            targets['returns']
        )

        # Combine losses with robust weighting
        total_loss = self.vol_weight * vol_loss + self.ret_weight * ret_loss

        # Create loss dictionary for logging
        loss_dict = {
            'vol_loss': vol_loss.item(),
            'ret_loss': ret_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    


class CalibratedConfidenceVolatilityLoss(nn.Module):
    """
    Implements a numerically stabilized heteroscedastic Gaussian NLL for volatility prediction
    with an extra calibration penalty that encourages the predicted confidence to reflect
    the actual error, not just the volatility regime.
    """
    def __init__(self, logvar_min=-3.0, logvar_max=3.0, eps=1e-6, reg_lambda=0.1, gamma=0.1):
        """
        Args:
            logvar_min, logvar_max: Limits to clamp log variance to avoid numerical issues.
            eps: A small constant to prevent division by zero.
            reg_lambda: Weight for the quadratic regularization on logvar (acts like a Gaussian prior).
            gamma: Weight for the additional calibration penalty.
        """
        super().__init__()
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.eps = eps
        self.reg_lambda = reg_lambda
        self.gamma = gamma  # New hyperparameter for calibration penalty.
        self.constant = 0.5 * math.log(2 * math.pi)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: Predicted volatility [batch, 1].
            logvar: Predicted log variance (confidence) [batch, 1].
            target: Actual volatility [batch, ...].
        """
        # Replace any NaN/Inf with safe values
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=self.logvar_max, neginf=self.logvar_min)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        # Standard heteroscedastic NLL components
        sq_error = (target - mu).pow(2)
        weighted_sq_error = 0.5 * torch.exp(-logvar) * sq_error
        log_var_term = 0.5 * logvar
        
        # Regularization on logvar to prevent extreme values
        reg_term = self.reg_lambda * logvar.pow(2)
        
        # --- Calibration Term ---
        # Compute the absolute error
        abs_error = torch.abs(target - mu)
        # Define a target confidence: when error is small, target_conf is high (near 1)
        # and when error is large, target_conf decreases.
        target_conf = 1.0 / (1.0 + abs_error + self.eps)
        # Predicted confidence is derived from logvar
        pred_conf = torch.exp(-logvar)
        # Penalize the absolute difference between the predicted and target confidence.
        calib_term = self.gamma * torch.mean(torch.abs(pred_conf - target_conf))
        
        # Combine all components and add a constant to complete the Gaussian log-likelihood.
        nll = weighted_sq_error + log_var_term + self.constant + reg_term + calib_term
        
        return nll.mean()
