import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class NaiveVolatilityModel:
    """
    Naive baseline model that uses the previous period's volatility as the prediction.
    """
    def __init__(self):
        self.name = "naive_baseline"
        self.params = {"description": "Using previous period volatility as prediction"}
        
    def fit(self, df: pd.DataFrame) -> None:
        """Nothing to fit for naive model."""
        pass
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using previous period's volatility.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Must contain 'Y_log_vol_10min_lag_1m' column
            
        Returns:
        --------
        pd.DataFrame with predictions added
        """
        result_df = df.copy()
        result_df['predicted_vol'] = df['Y_log_vol_10min_lag_1m']
        return result_df


def train_and_validate_naive(train_df: pd.DataFrame, 
                           val_df: pd.DataFrame,
                           evaluator) -> Tuple[NaiveVolatilityModel, Dict]:
    """
    Train and validate the naive baseline model.
    
    Parameters:
    -----------
    train_df : Training data
    val_df : Validation data
    evaluator : ModelEvaluator instance
    
    Returns:
    --------
    Tuple of (fitted model, validation metrics)
    """
    model = NaiveVolatilityModel()
    
    # No fitting needed for naive model
    model.fit(train_df)
    
    # Generate predictions
    train_pred = model.predict(train_df)
    val_pred = model.predict(val_df)
    
    # Evaluate train set
    train_metrics = evaluator.evaluate_predictions(train_pred)
    evaluator.log_results(
        model_name=model.name,
        metrics=train_metrics,
        params=model.params,
        dataset_type="train",
        notes="Initial naive baseline evaluation"
    )
    
    # Evaluate validation set
    val_metrics = evaluator.evaluate_predictions(val_pred)
    evaluator.log_results(
        model_name=model.name,
        metrics=val_metrics,
        params=model.params,
        dataset_type="validate",
        notes="Initial naive baseline evaluation"
    )
    
    return model, val_metrics


def evaluate_final_model(model: NaiveVolatilityModel,
                        test_df: pd.DataFrame,
                        evaluator) -> Dict:
    """
    Evaluate the final model on test set. Should only be called once!
    
    Parameters:
    -----------
    model : Trained model
    test_df : Test data
    evaluator : ModelEvaluator instance
    
    Returns:
    --------
    Dict containing test metrics
    """
    test_pred = model.predict(test_df)
    test_metrics = evaluator.evaluate_predictions(test_pred)
    
    evaluator.log_results(
        model_name=model.name,
        metrics=test_metrics,
        params=model.params,
        dataset_type="test",
        notes="FINAL TEST SET EVALUATION - DO NOT REPEAT"
    )
    
    return test_metrics