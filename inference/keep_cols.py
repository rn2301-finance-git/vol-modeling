# Define columns to keep
KEEP_COLUMNS = [
    # Identifiers and timestamps
    'minute', 'symbol', 'ts_event',
    
    # Market data
    'mid', 'spread', 'imbalance',
    
    'roll_vol_5m','roll_vol_10m','roll_vol_30m','roll_vol_60m',
    
    'vol_of_vol_60m',
    
    # Returns (both with and without lags)
    'Y_log_ret_10min', 'Y_log_ret_30min', 'Y_log_ret_60min',
    'Y_log_ret_10min_lag_1m', 'Y_log_ret_30min_lag_1m', 'Y_log_ret_60min_lag_1m',
    
    # Volatility (both with and without lags)
    'roll_vol_10m', 
    'Y_log_vol_10min', 'Y_log_vol_10min_lag_1m',
    
    # Validity flags
    'valid_target_10m', 'valid_target_30m', 'valid_target_60m',
    'Y_log_ret_10min_partial', 'Y_log_ret_30min_partial', 'Y_log_ret_60min_partial',
    
]