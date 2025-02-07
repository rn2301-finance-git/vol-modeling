import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.optimize import minimize
from typing import Optional, List, Dict, Tuple, Any
import logging
import torch

# Re-use the dataset logic from deep_learning_model.py
from models.deep_learning_model import VolatilityDataset

# Re-use your evaluator and run_manager
from evaluation.evaluator import ModelEvaluator
from evaluation.run_manager import RunManager


class AsymmetricLasso(BaseEstimator, RegressorMixin):
    """
    Custom Lasso implementation with an asymmetric squared-error loss
    plus L1 regularization. Uses L-BFGS-B for optimization.

    Parameters
    ----------
    alpha : float
        L1 regularization strength.
    over_penalty : float
        Multiplier for squared residuals when residual >= 0.
    under_penalty : float
        Multiplier for squared residuals when residual < 0.
    max_iter : int
        Maximum iterations for the optimizer.
    tol : float
        Tolerance for the optimizer.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        over_penalty: float = 1.0,
        under_penalty: float = 2.0,
        max_iter: int = 1000,
        tol: float = 1e-4
    ):
        self.alpha = alpha
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = 0.0
        self.feature_names_ = None

    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Scalar objective = mean(asymmetric squared residual) + alpha * L1(...) on coef.

        Asymmetric squared residual:
            If residual >= 0: over_penalty * residual^2
            If residual <  0: under_penalty * residual^2
        """
        coef = params[:-1]
        intercept = params[-1]
        preds = X @ coef + intercept

        residual = preds - y
        mask = (residual >= 0)  # True if residual >= 0
        # Weighted squared errors
        sq_errors = np.where(
            mask,
            self.over_penalty * residual**2,
            self.under_penalty * residual**2
        )
        data_loss = np.mean(sq_errors)

        # L1 penalty on coefficients (exclude intercept)
        l1_penalty = self.alpha * np.sum(np.abs(coef))

        return data_loss + l1_penalty

    def _gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of the objective w.r.t. all parameters [coef, intercept].

        d/dw_j of data_loss + alpha * sum(|w_j|)
        where data_loss is the mean of asymmetric squared residuals.
        """
        coef = params[:-1]
        intercept = params[-1]
        preds = X @ coef + intercept
        residual = preds - y

        # Identify where residual >= 0 vs < 0
        mask = (residual >= 0)
        n_samples = X.shape[0]

        # derivative of "over_penalty * residual^2" w.r.t residual is 2*over_penalty*residual
        # derivative of "under_penalty * residual^2" w.r.t residual is 2*under_penalty*residual
        grad_wrt_pred = np.where(
            mask,
            2.0 * self.over_penalty * residual,
            2.0 * self.under_penalty * residual
        ) / n_samples  # divide by N for mean

        # Gradient w.r.t. coef: X^T * grad_wrt_pred
        grad_coef = X.T @ grad_wrt_pred

        # Add L1 subgradient: alpha * sign(w_j)
        # Note: subgradient at w_j=0 is in [-alpha, alpha], here we use sign(0) = 0
        grad_coef += self.alpha * np.sign(coef)

        # Gradient w.r.t. intercept is sum of grad_wrt_pred
        grad_intercept = np.sum(grad_wrt_pred)

        return np.concatenate([grad_coef, [grad_intercept]])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AsymmetricLasso':
        """
        Fit model using scipy.optimize.minimize with L-BFGS-B.
        """
        n_features = X.shape[1]
        initial_params = np.zeros(n_features + 1)  # [coefs..., intercept]

        # Run optimization
        result = minimize(
            fun=lambda p: self._objective(p, X, y),
            x0=initial_params,
            jac=lambda p: self._gradient(p, X, y),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )

        if not result.success:
            logging.warning(f"Optimization failed to converge: {result.message}")

        # Extract fitted params
        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions given input X.
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.coef_ + self.intercept_

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on coefficient magnitudes.
        """
        if self.coef_ is None or self.feature_names_ is None:
            raise ValueError("Model must be fitted with feature names.")
        return dict(zip(self.feature_names_, np.abs(self.coef_)))


def train_lasso_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Y_log_vol_10min_lag_1m",
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
    use_cuda: bool = False,  # Kept for API consistency
    run_manager: Optional[RunManager] = None,
    asymmetric_loss: bool = False
) -> Tuple[AsymmetricLasso, StandardScaler]:
    """
    Train a (potentially) asymmetric Lasso model with interface matching your other models.
    
    If asymmetric_loss=True, uses over_penalty=1.0, under_penalty=2.0.
    Otherwise, it defaults to a symmetric MSE (over_penalty=under_penalty=1.0).
    """
    # 1) Log config if run_manager provided
    if run_manager:
        model_params = {
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'device': 'cpu',
            'asymmetric_loss': asymmetric_loss
        }
        dataset_info = {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'feature_count': len(feature_cols),
            'feature_names': feature_cols
        }
        run_manager.save_config(
            args={},
            dataset_info=dataset_info,
            model_params=model_params
        )
        run_manager.logger.info("Training Lasso on CPU")

    # 2) Prepare data using VolatilityDataset (same approach as XGBoost or MLP)
    train_dataset = VolatilityDataset(train_df, feature_cols, target_col, fit_target=True)
    target_scaler = train_dataset.get_target_scaler()
    val_dataset = VolatilityDataset(val_df, feature_cols, target_col, target_scaler, fit_target=False)

    X_train = train_dataset.X.numpy()
    y_train = train_dataset.y.numpy()
    X_val = val_dataset.X.numpy()
    y_val = val_dataset.y.numpy()

    # Decide on over_penalty, under_penalty
    if asymmetric_loss:
        over_penalty, under_penalty = 1.0, 2.0
    else:
        over_penalty, under_penalty = 1.0, 1.0

    # 3) Create and fit the AsymmetricLasso model
    model = AsymmetricLasso(
        alpha=alpha,
        over_penalty=over_penalty,
        under_penalty=under_penalty,
        max_iter=max_iter,
        tol=tol
    )
    model.feature_names_ = feature_cols
    model.fit(X_train, y_train)

    # 4) Log metrics if run_manager provided
    if run_manager:
        # Get predictions for both train and validation sets
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val) if len(X_val) > 0 else None

        # Calculate detailed metrics
        metrics = {}
        
        # Training metrics
        train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
        train_rank_corr = np.corrcoef(
            y_train, train_preds, rowvar=False
        )[0, 1] if len(train_preds) > 1 else 0.0
        
        metrics.update({
            'train_rmse': train_rmse,
            'train_rank_corr': train_rank_corr,
            'train_mse': train_rmse ** 2,
            'feature_importance': model.get_feature_importance()
        })

        # Validation metrics
        if val_preds is not None:
            val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
            val_rank_corr = np.corrcoef(
                y_val, val_preds, rowvar=False
            )[0, 1] if len(val_preds) > 1 else 0.0
            
            # Calculate asymmetric MSE if requested
            if asymmetric_loss:
                residuals = val_preds - y_val
                over_pred_mask = residuals >= 0
                val_asymm_mse = np.mean(
                    np.where(over_pred_mask, 
                            2.0 * residuals**2,  # Over-predictions penalized more
                            residuals**2)
                )
                metrics['val_asymm_mse'] = val_asymm_mse

            metrics.update({
                'val_rmse': val_rmse,
                'val_rank_corr': val_rank_corr,
                'val_mse': val_rmse ** 2
            })

        run_manager.log_metrics(metrics, step=max_iter)

        # Save model checkpoint
        run_manager.save_checkpoint(
            model={
                'coef_': model.coef_,
                'intercept_': model.intercept_,
                'feature_names_': model.feature_names_
            },
            optimizer=None,
            epoch=-1,
            val_loss=metrics.get('val_rmse', float('inf')),
            val_metrics=metrics,
            is_best=True
        )

    return model, target_scaler


def predict_lasso_model(
    model: AsymmetricLasso,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_scaler: StandardScaler,
    target_col: str = "Y_log_vol_10min_lag_1m"
) -> pd.DataFrame:
    """
    Generate predictions using trained Lasso model, 
    inverse-transforming the target for final predictions.
    """
    # Use VolatilityDataset for consistent scaling
    temp_dataset = VolatilityDataset(df, feature_cols, target_col, target_scaler, fit_target=False)
    X_data = temp_dataset.X.numpy()

    # Get scaled predictions
    preds_scaled = model.predict(X_data)

    # Inverse transform
    preds_unscaled = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()
    final_preds = np.expm1(preds_unscaled)  # if you originally log1p-transformed the target

    df_copy = df.copy()
    df_copy["predicted_vol"] = final_preds
    return df_copy


def evaluate_lasso_model(
    model: AsymmetricLasso,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    evaluator: ModelEvaluator,
    target_scaler: StandardScaler,
    feature_cols: List[str],
    target_col: str = "Y_log_vol_10min_lag_1m",
    notes: str = ""
) -> Dict[str, float]:
    """
    Evaluate trained Lasso model using ModelEvaluator.
    Logs metrics for both training and validation sets.
    """
    # 1) Evaluate on training set
    train_pred_df = predict_lasso_model(
        model=model,
        df=train_df,
        feature_cols=feature_cols,
        target_scaler=target_scaler,
        target_col=target_col
    )
    train_metrics = evaluator.evaluate_predictions(
        train_pred_df,
        y_true_col=target_col,
        y_pred_col="predicted_vol"
    )
    evaluator.log_results(
        model_name="lasso_model",
        metrics=train_metrics,
        params={
            "model_type": "Lasso",
            "notes": notes,
            "feature_importance": model.get_feature_importance()
        },
        dataset_type="train"
    )

    # 2) Evaluate on validation set
    val_metrics = None
    if len(val_df) > 0:
        val_pred_df = predict_lasso_model(
            model=model,
            df=val_df,
            feature_cols=feature_cols,
            target_scaler=target_scaler,
            target_col=target_col
        )
        val_metrics = evaluator.evaluate_predictions(
            val_pred_df,
            y_true_col=target_col,
            y_pred_col="predicted_vol"
        )
        evaluator.log_results(
            model_name="lasso_model",
            metrics=val_metrics,
            params={
                "model_type": "Lasso",
                "notes": notes,
                "feature_importance": model.get_feature_importance()
            },
            dataset_type="validate"
        )

    # 3) Print latest results summary
    evaluator.print_latest_summary(n_latest=2)
    
    return val_metrics if val_metrics is not None else {}
