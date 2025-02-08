# xgboost_model.py
import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Dict
from xgboost import XGBRegressor
import xgboost

# Re-use the dataset/scaler logic from deep_learning_model.py
from models.deep_learning_model import VolatilityDataset

# Re-use your evaluator and run_manager
from evaluation.evaluator import ModelEvaluator
from evaluation.run_manager import RunManager, setup_logging

# Get logger for this module
logger = setup_logging().getChild('xgboost')

def asymmetric_loss_objective(preds, dtrain, alpha=1.5):
    """
    Custom asymmetric loss function for XGBoost that penalizes under-predictions more than over-predictions.
    
    Parameters
    ----------
    preds : np.ndarray
        Current predictions
    dtrain : xgboost.DMatrix
        Training data
    alpha : float
        Factor to penalize under-predictions (default: 1.5)
        
    Returns
    -------
    tuple
        Gradient and Hessian for XGBoost optimization
    """
    labels = dtrain.get_label()
    residual = preds - labels
    
    grad = np.where(residual > 0, residual, alpha * residual)
    hess = np.ones_like(grad)
    
    return grad, hess

def asymmetric_loss_eval(preds, dtrain, alpha=1.5):
    """
    Evaluation metric for asymmetric loss that matches the objective function.
    
    Parameters
    ----------
    preds : np.ndarray
        Current predictions
    dtrain : xgboost.DMatrix
        Training data
    alpha : float
        Factor to penalize under-predictions
        
    Returns
    -------
    tuple
        Name of metric and computed value
    """
    labels = dtrain.get_label()
    residual = preds - labels
    
    # Calculate asymmetric loss
    loss = np.where(residual > 0, residual**2, (alpha * residual)**2)
    return 'asymm_loss', np.mean(loss)

def train_xgboost_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Y_log_vol_10min_lag_1m",
    target: str = "vol",  # New parameter: "vol" or "ret"
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    use_cuda: bool = False,
    run_manager: Optional[RunManager] = None,
    asymmetric_loss: bool = False,
    asymmetric_alpha: float = 1.5,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
):
    """
    Train an XGBoost model.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset
    val_df : pd.DataFrame
        Validation dataset
    feature_cols : List[str]
        List of feature column names
    target_col : str
        Name of the target column (default "Y_log_vol_10min_lag_1m")
    target : str
        Target type ("vol" for volatility, "ret" for returns)
    n_estimators : int
        Number of boosting rounds (default 200)
    learning_rate : float
        Learning rate (default 0.1)
    max_depth : int
        Max tree depth (default 6)
    subsample : float
        Subsample ratio of the training instances (default 0.8)
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree (default 0.8)
    use_cuda : bool
        Whether to use GPU acceleration (requires GPU + xgboost built with GPU support)
    run_manager : Optional[RunManager]
        If provided, logs config/metrics/checkpoints to your run manager
    asymmetric_loss : bool
        Whether to use asymmetric loss function that penalizes under-predictions more (default False)
    asymmetric_alpha : float
        Factor to penalize under-predictions when using asymmetric loss (default 1.5)
    reg_alpha : float
        L1 regularization term on weights (default: 0.0)
    reg_lambda : float 
        L2 regularization term on weights (default: 1.0)
    gamma : float
        Minimum loss reduction required to make a further partition (default: 0.0)
    
    Returns
    -------
    model : XGBRegressor
        Trained XGBoost model
    target_scaler : StandardScaler (or None)
        The target scaler used to invert predictions (same logic as MLP). 
    """
    # Always use CPU for XGBoost
    device = "cpu"
    
    # Set target column based on target parameter
    if target == "ret":
        target_col = "Y_log_ret_60min_lag_1m"
        logger.info("Training XGBoost model for returns prediction")
    else:  # default to "vol"
        target_col = "Y_log_vol_10min_lag_1m"
        logger.info("Training XGBoost model for volatility prediction")
    
    logger.info("\n" + "="*80)
    logger.info("Starting XGBoost Training")
    logger.info("="*80)
    
    # Log dataset info
    logger.info(f"\nDataset Information:")
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Validation samples: {len(val_df):,}")
    logger.info(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    logger.info(f"Target column: {target_col}")
    
    # Log model parameters
    logger.info("\nModel Parameters:")
    logger.info(f"n_estimators: {n_estimators}")
    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"max_depth: {max_depth}")
    logger.info(f"subsample: {subsample}")
    logger.info(f"colsample_bytree: {colsample_bytree}")
    logger.info(f"asymmetric_loss: {asymmetric_loss}")
    if asymmetric_loss:
        logger.info(f"asymmetric_alpha: {asymmetric_alpha}")
    logger.info(f"reg_alpha: {reg_alpha}")
    logger.info(f"reg_lambda: {reg_lambda}")
    logger.info(f"gamma: {gamma}")
    
    try:
        # 1) Log config to RunManager, if available
        if run_manager:
            model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'gamma': gamma,
                'tree_method': "hist",
                'device': device
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
            run_manager.logger.info(f"Training XGBoost on CPU")
        
        # 2) Prepare scaled/log1p data for train/val
        train_dataset = VolatilityDataset(train_df, feature_cols, target_col, fit_target=True)
        target_scaler = train_dataset.get_target_scaler()
        val_dataset = VolatilityDataset(val_df, feature_cols, target_col, target_scaler, fit_target=False)
        
        X_train = train_dataset.X.numpy()
        y_train = train_dataset.y.numpy()
        X_val = val_dataset.X.numpy()
        y_val = val_dataset.y.numpy()
        
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        
        # Log data statistics
        logger.info("\nData Statistics:")
        logger.info(f"X_train - mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")
        logger.info(f"y_train - mean: {y_train.mean():.3f}, std: {y_train.std():.3f}")
        if len(X_val) > 0:
            logger.info(f"X_val - mean: {X_val.mean():.3f}, std: {X_val.std():.3f}")
            logger.info(f"y_val - mean: {y_val.mean():.3f}, std: {y_val.std():.3f}")
        
        # 3) Create and fit XGBRegressor
        model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "gamma": gamma,
            "tree_method": "hist",
            "device": "cpu",
            "missing": np.nan,
            "eval_metric": 'rmse'
        }
        
        if asymmetric_loss:
            # For asymmetric loss, we need to use the lower-level XGBoost API
            dtrain = xgboost.DMatrix(X_train, label=y_train)
            if len(X_val) > 0:
                dval = xgboost.DMatrix(X_val, label=y_val)
                eval_list = [(dtrain, 'train'), (dval, 'validation_0')]
            else:
                eval_list = [(dtrain, 'train')]
            
            # Add custom evaluation metric
            feval = lambda preds, dtrain: asymmetric_loss_eval(preds, dtrain, alpha=asymmetric_alpha)
            
            # Train with custom objective and configured alpha
            model = xgboost.train(
                params=model_params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=eval_list,
                obj=lambda preds, dtrain: asymmetric_loss_objective(preds, dtrain, alpha=asymmetric_alpha),
                feval=feval,  # Add custom evaluation metric
                verbose_eval=False
            )
            
            if run_manager:
                # Get predictions for metrics
                train_preds = model.predict(dtrain)
                
                # Calculate training metrics using evaluator
                train_metrics = evaluator.evaluate_raw_arrays(
                    predictions=train_preds,
                    targets=y_train
                )
                
                final_metrics = {
                    'train_rmse': train_metrics['rmse'],
                    'train_rank_corr': train_metrics['rank_corr'],
                    'train_asymm_mse': train_metrics.get('asymm_mse', None),
                    'feature_importance': dict(zip(feature_cols, model.get_score(importance_type='weight'))),
                    'total_rounds': n_estimators
                }
                
                # Calculate validation metrics if validation data exists
                if len(X_val) > 0:
                    val_preds = model.predict(dval)
                    val_metrics = evaluator.evaluate_raw_arrays(
                        predictions=val_preds,
                        targets=y_val
                    )
                    
                    final_metrics.update({
                        'val_rmse': val_metrics['rmse'],
                        'val_rank_corr': val_metrics['rank_corr'],
                        'val_asymm_mse': val_metrics.get('asymm_mse', None)
                    })
                
                # Save to final_metrics.json
                run_manager.save_final_metrics(final_metrics)
        else:
            # Regular XGBRegressor path needs updating
            model = XGBRegressor(**model_params)
            
            # Fit model
            if len(X_val) > 0:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                if run_manager:
                    # Calculate train predictions and metrics
                    train_preds = model.predict(X_train)
                    train_rmse = np.sqrt(np.mean((y_train - train_preds)**2))
                    train_rank_corr = np.corrcoef(train_preds, y_train)[0,1]
                    
                    # Calculate validation predictions and metrics
                    val_preds = model.predict(X_val)
                    val_rmse = np.sqrt(np.mean((y_val - val_preds)**2))
                    val_rank_corr = np.corrcoef(val_preds, y_val)[0,1]
                    
                    # Log final metrics
                    final_metrics = {
                        'train_rmse': train_rmse,
                        'train_rank_corr': train_rank_corr,
                        'val_rmse': val_rmse,
                        'val_rank_corr': val_rank_corr,
                        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
                    }
                    run_manager.log_metrics(final_metrics, n_estimators)
                    
                    # Save XGBoost model using the specific XGBoost save method
                    run_manager.save_xgboost_checkpoint(
                        model=model,
                        val_metrics=final_metrics,
                        is_best=True
                    )
            else:
                model.fit(X_train, y_train)
        
        logger.info("\nTraining completed successfully!")
        
        # Log feature importances
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(feature_cols, model.feature_importances_))
            logger.info("\nTop 10 Feature Importances:")
            for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"{feat}: {imp:.4f}")
        
        return model, target_scaler
        
    except Exception as e:
        logger.error(f"Error during XGBoost training: {str(e)}", exc_info=True)
        raise


def predict_xgboost_model(
    model,  # Changed type hint since it could be either XGBRegressor or Booster
    df: pd.DataFrame,
    feature_cols: List[str],
    target_scaler,
    target_col: str = "Y_log_vol_10min_lag_1m"
) -> pd.DataFrame:
    """
    Generate predictions for a given DataFrame using the trained XGBoost model.
    Works with both XGBRegressor and lower-level Booster models.
    """
    logger.info("\nGenerating predictions...")
    logger.info(f"Input data shape: {df.shape}")
    
    try:
        # Create a small dataset to use the same scaling logic
        temp_dataset = VolatilityDataset(df, feature_cols, target_col, target_scaler, fit_target=False)
        X_data = temp_dataset.X.numpy()
        
        logger.info(f"Processed features shape: {X_data.shape}")
        logger.info(f"Feature statistics - mean: {X_data.mean():.3f}, std: {X_data.std():.3f}")
        
        # Handle both XGBRegressor and Booster models
        if isinstance(model, XGBRegressor):
            preds_scaled = model.predict(X_data)
        else:  # Booster model from xgboost.train()
            dmatrix = xgboost.DMatrix(X_data)
            preds_scaled = model.predict(dmatrix)
        
        # Invert StandardScaler
        preds_unscaled = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()
        # Invert log1p
        final_preds = np.expm1(preds_unscaled)
        
        logger.info("\nPrediction Statistics:")
        logger.info(f"Scaled predictions - mean: {preds_scaled.mean():.3f}, std: {preds_scaled.std():.3f}")
        logger.info(f"Final predictions - mean: {final_preds.mean():.3f}, std: {final_preds.std():.3f}")
        
        df_copy = df.copy()
        df_copy["predicted_vol"] = final_preds
        return df_copy
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise


def evaluate_xgboost_model(
    model: XGBRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    evaluator: ModelEvaluator,
    target_scaler,
    feature_cols: List[str],
    target_col: str = "Y_log_vol_10min_lag_1m",
    notes: str = ""
) -> Dict:
    """
    Evaluate a trained XGBoost model on train/validation sets using ModelEvaluator.
    
    Returns:
    --------
    Dict
        Dictionary containing at minimum:
        - rmse: float
        - rank_corr: float
        - asymm_mse: float
    """
    try:
        # 1) Predict on train
        train_pred_df = predict_xgboost_model(
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
            model_name="xgboost_model",
            metrics=train_metrics,
            params={"model_type": "XGBoost", "notes": notes},
            dataset_type="train"
        )
        
        # 2) Predict on val and return val metrics
        if len(val_df) > 0:
            val_pred_df = predict_xgboost_model(
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
                model_name="xgboost_model",
                metrics=val_metrics,
                params={"model_type": "XGBoost", "notes": notes},
                dataset_type="validate"
            )
            
            # Ensure we return at least the required metrics
            return {
                "rmse": val_metrics.get("rmse", float('inf')),
                "rank_corr": val_metrics.get("rank_corr", 0.0),
                "asymm_mse": val_metrics.get("asymm_mse", float('inf')),
                **val_metrics  # Include any additional metrics
            }
        else:
            # If no validation data, return train metrics
            return {
                "rmse": train_metrics.get("rmse", float('inf')),
                "rank_corr": train_metrics.get("rank_corr", 0.0),
                "asymm_mse": train_metrics.get("asymm_mse", float('inf')),
                **train_metrics  # Include any additional metrics
            }
            
    except Exception as e:
        logger.error(f"Error in evaluate_xgboost_model: {str(e)}", exc_info=True)
        # Return default metrics that match what run_experiment expects
        return {
            "rmse": float('inf'),
            "rank_corr": 0.0,
            "asymm_mse": float('inf')
        }
