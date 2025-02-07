lasso_features = [
    # Core price & liquidity metrics
    'mid', 'spread', 'imbalance', 'log_turnover',
    
    # Market microstructure
    'bid_sz_00', 'ask_sz_00', 'bid_ct_00', 'ask_ct_00',
    
    # Returns at key timeframes
    'log_ret_1m', 'log_ret_5m', 'log_ret_30m',
    
    # Rolling volatility metrics
    'roll_vol_5m', 'roll_vol_30m',
    
    # EWMA features
    'ewma_spread_10', 'ewma_imbalance_10',
    
    # Time features
    'normalized_time', 'time_cos', 'time_sin',
    
    # Price dynamics
    'range_5m', 'range_30m',
    'vol_of_vol_5m', 'vol_of_vol_30m',
    'price_accel_5m', 'price_accel_30m',
    
    # Moving averages and volatility
    'spread_ma_10m', 'spread_vol',
    'imbalance_ma_10m', 'imbalance_vol',
    
    # Technical indicators
    'rsi', 'bb_ma', 'bb_std', 'bb_width',
    
    # Higher moments
    'skew_30m', 'kurt_30m',
    
    # Daily features
    'avg_vol_1d', 'avg_vol_5d',
    
    # Cross-sectional features
    'avg_nn_roll_vol_5m', 'avg_nn_roll_vol_30m',
    'avg_nn_log_ret_5m', 'avg_nn_log_ret_30m'
]

xg_dl_features = [
    # All price related
    'price', 'size', 'mid', 'spread', 'imbalance',
    'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00',
    'bid_ct_00', 'ask_ct_00', 'log_turnover',
    
    
    # All return timeframes
    'log_ret_1m', 'log_ret_2m', 'log_ret_5m', 'log_ret_10m',
    'log_ret_30m', 'log_ret_60m',
    
    # EWMA features
    'ewma_spread_10', 'ewma_imbalance_10',
    
    # All volatility timeframes
    'roll_vol_5m', 'roll_vol_10m', 'roll_vol_30m', 'roll_vol_60m',
    
    # Time features
    'normalized_time', 'time_cos', 'time_sin',
    
    # All ranges and dynamics
    'range_5m', 'range_10m', 'range_30m', 'range_60m',
    'vol_of_vol_5m', 'vol_of_vol_10m', 'vol_of_vol_30m', 'vol_of_vol_60m',
    'price_accel_5m', 'price_accel_10m', 'price_accel_30m', 'price_accel_60m',
    
    # Market microstructure
    'spread_ma_10m', 'spread_vol',
    'imbalance_ma_10m', 'imbalance_vol',
    
    # Technical indicators
    'rsi', 'bb_ma', 'bb_std', 'bb_width',
    
    # All higher moments
    'skew_10m', 'kurt_10m', 'skew_30m', 'kurt_30m',
    'skew_60m', 'kurt_60m',
    
    # Time features
    'minute_offset',
    
    # All daily features
    'avg_vol_1d', 'avg_vol_2d', 'avg_vol_5d', 'avg_vol_10d',
    
    # All cross-sectional features
    'avg_nn_roll_vol_5m', 'avg_nn_roll_vol_10m',
    'avg_nn_roll_vol_30m', 'avg_nn_roll_vol_60m',
    'avg_nn_log_ret_1m', 'avg_nn_log_ret_2m',
    'avg_nn_log_ret_5m', 'avg_nn_log_ret_10m',
    'avg_nn_log_ret_30m', 'avg_nn_log_ret_60m'
]


selected_features = [
    'mid',                  # Clean price signal
    'spread',              # Liquidity indicator
    'log_ret_1m',          # Most recent return
    'log_ret_5m',          # Medium-term return
    'imbalance',           # Order book imbalance
    'log_turnover',        # Trading activity
] + [
    'roll_vol_5m',         # Short-term volatility
    'roll_vol_30m',        # Longer volatility (skip 10m to avoid target leakage)
    'vol_of_vol_5m',       # Volatility regime indicator
    'vol_regime',          # From your daily metrics
    'rel_vol',             # Stock vol vs market
    'vol_rank'             # Cross-sectional position
] + [
    'bid_sz_00',          # Top of book depth
    'ask_sz_00',          # Top of book depth
    'spread_ma_10m',      # Liquidity trend
    'imbalance_ma_10m'    # Order flow trend
] + [
    'normalized_time',     # Position in trading day
    'time_cos',           # Circular time feature
    'time_sin'            # Circular time feature
] + [
    'avg_nn_roll_vol_5m',  # Peer group volatility
    'avg_nn_log_ret_5m'    # Peer group returns
]