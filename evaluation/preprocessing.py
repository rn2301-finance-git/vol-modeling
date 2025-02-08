import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import logging
import time
import s3fs
from data_pipeline.features_in_model import selected_features
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from data_pipeline.sequence_dataset import SequenceVolatilityDataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
import pyarrow as pa
import pandas_market_calendars as mcal
from data_pipeline.sequence_dataset import ThreeHeadedTransformerDataset
from evaluation.run_manager import setup_logging
import os

# Define dataset splits
SPLITS = {
    'train': ('2024-01-16', '2024-06-30'),
    'validate': ('2024-07-01', '2024-08-31'), 
    'test': ('2024-09-01', '2025-01-14')
}

# Define test mode splits (10 days each)
TEST_SPLITS = {
    'train': ('2024-03-01', '2024-03-10'),
    'validate': ('2024-03-11', '2024-03-20'),
    'test': ('2024-03-21', '2024-03-30')
}

# Get logger for this module
logger = setup_logging().getChild('preprocessing')

# Add after imports
BUCKET_NAME = os.environ.get('BUCKET_NAME')
if not BUCKET_NAME:
    raise ValueError("Environment variable BUCKET_NAME must be set")

def handle_missing_values(df: pd.DataFrame, feature_cols: List[str], model_type: str = None) -> pd.DataFrame:
    """
    Preprocess features with model-specific missing value handling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features
    feature_cols : List[str]
        List of feature column names to process
    model_type : str
        Type of model being trained ('xgboost', 'mlp', etc.)
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with handled missing values
    """
    # Sort for consistent processing
    df = df.sort_values(['symbol', 'date', 'minute'])
    
    # Replace infinities with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    if model_type == 'xgboost':
        logger.info("Using XGBoost - keeping NaN values as-is")
        return df

    # For non-XGBoost models, proceed with original imputation logic
    # Create missing value flags
    was_missing = df[feature_cols].isna().astype(int).add_suffix('_was_missing')
    df = pd.concat([df, was_missing], axis=1)

    # Forward fill and median imputation steps...
    # 2. Forward fill within each (symbol, date) group
    df[feature_cols] = (
        df.groupby(['symbol', 'date'], group_keys=False)[feature_cols]
          .ffill()
    )

    # 3. Fill remaining NaNs with symbol-level medians
    df[feature_cols] = (
        df.groupby('symbol', group_keys=False)[feature_cols]
          .apply(lambda g: g.fillna(g.median()))
    )

    # 4. Final fallback to global median for any remaining NaNs
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    return df

def preprocess_data(df: pd.DataFrame, feature_cols: List[str], model_type: str = None) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Scale features using StandardScaler with enhanced validation, unless using XGBoost.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features
    feature_cols : List[str]
        List of feature column names to process
    model_type : str
        Type of model being trained ('xgboost', 'mlp', etc.)
    
    Returns:
    --------
    Tuple[pd.DataFrame, Optional[StandardScaler]]
        Processed dataframe and fitted scaler (None if XGBoost)
    """
    logger.info("\nPre-processing statistics:")
    pre_stats = df[feature_cols].agg(['min', 'max', 'mean', 'std']).round(4)
    for feature in feature_cols:
        logger.info(f"\n{feature}:")
        logger.info(f"  min:  {pre_stats.loc['min', feature]:>10.4f}")
        logger.info(f"  max:  {pre_stats.loc['max', feature]:>10.4f}")
        logger.info(f"  mean: {pre_stats.loc['mean', feature]:>10.4f}")
        logger.info(f"  std:  {pre_stats.loc['std', feature]:>10.4f}")
    
    if model_type == 'xgboost':
        logger.info("\nSkipping StandardScaler for XGBoost model")
        return df, None
    
    # Fit and transform for non-XGBoost models
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    # Validate scaling results
    logger.info("\nPost-scaling statistics:")
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
    post_scale_stats = scaled_df.agg(['min', 'max', 'mean', 'std']).round(4)
    
    logger.info("\nFeature-wise statistics:")
    for feature in feature_cols:
        logger.info(f"\n{feature}:")
        logger.info(f"  min:  {post_scale_stats.loc['min', feature]:>10.4f}")
        logger.info(f"  max:  {post_scale_stats.loc['max', feature]:>10.4f}")
        logger.info(f"  mean: {post_scale_stats.loc['mean', feature]:>10.4f}")
        logger.info(f"  std:  {post_scale_stats.loc['std', feature]:>10.4f}")
    
    # Store scaling parameters for reference
    scaler.feature_names = feature_cols
    scaler.scale_stats = {
        'pre': pre_stats.to_dict(),
        'post': post_scale_stats.to_dict()
    }
    
    df[feature_cols] = scaled_features
    return df, scaler

def create_sequence_datasets(
    datasets: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    short_seq_len: int = 20,
    long_seq_len: int = 60,
    sample_every: int = 10,
    target_col: str = "Y_log_vol_10min",
    use_lagged_targets: bool = True
) -> Dict[str, DataLoader]:
    """
    Create sequence datasets from preprocessed DataFrames.
    
    Parameters:
    -----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of DataFrames for each split
    feature_cols : List[str]
        List of feature column names
    short_seq_len : int
        Length of short sequence window
    long_seq_len : int
        Length of long sequence window
    sample_every : int
        Sample sequences every N steps
    target_col : str
        Name of target column
    use_lagged_targets : bool
        Whether to use lagged targets (default: True)
        
    Returns:
    --------
    Dict[str, DataLoader]
        Dictionary of DataLoaders for each split
    """
    logger = logging.getLogger('model_training')
    sequence_loaders = {}
    
    # Validate target column matches lagged/non-lagged setting
    if use_lagged_targets and "lag" not in target_col:
        raise ValueError(f"Using lagged targets but target column {target_col} does not contain 'lag'")
    if not use_lagged_targets and "lag" in target_col:
        raise ValueError(f"Not using lagged targets but target column {target_col} contains 'lag'")
    
    logger.info(f"\nCreating sequence datasets with target column: {target_col}")
    logger.info(f"Using {'lagged' if use_lagged_targets else 'non-lagged'} targets")

    for split, df in datasets.items():
        dataset = SequenceVolatilityDataset(
            df=df,
            short_seq_len=short_seq_len,
            long_seq_len=long_seq_len,
            sample_every=sample_every,
            feature_cols=feature_cols,
            target_col=target_col
        )
        
        shuffle = (split == 'train')  
        sequence_loaders[split] = DataLoader(
            dataset,
            batch_size=2048,  # Increased from 64 for better throughput
            shuffle=shuffle,
            num_workers=8,    # Increased from 4 for faster data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4  # Added to prefetch batches in advance
        )
        
        logger.info(f"Created DataLoader for {split} split with {len(dataset)} sequences")
    
    return sequence_loaders

def create_transformer_datasets(
    datasets: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    sequence_length: int = 60,
    sample_every: int = 10,
    vol_target_col: str = "Y_log_vol_10min",
    ret_target_col: str = "Y_log_ret_60min",
    model_type: Optional[str] = None,
    use_lagged_targets: bool = False
) -> Dict[str, DataLoader]:
    """
    Create transformer datasets from preprocessed DataFrames.
    
    Parameters:
    -----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of DataFrames for each split
    feature_cols : List[str]
        List of feature column names
    sequence_length : int
        Length of sequences to create
    sample_every : int
        Sampling interval for sequences
    vol_target_col : str
        Name of volatility target column
    ret_target_col : str
        Name of returns target column
    model_type : Optional[str]
        Type of model being trained (for missing value handling)
    use_lagged_targets : bool
        Whether to use lagged targets (default: False)
    """
    logger = logging.getLogger('model_training')
    transformer_loaders = {}
    
    # Validate target columns match lagged/non-lagged setting
    if use_lagged_targets:
        if "lag" not in vol_target_col or "lag" not in ret_target_col:
            raise ValueError(f"Using lagged targets but target columns {vol_target_col}, {ret_target_col} do not both contain 'lag'")
    else:
        if "lag" in vol_target_col or "lag" in ret_target_col:
            raise ValueError(f"Not using lagged targets but target columns {vol_target_col}, {ret_target_col} contain 'lag'")
    
    logger.info(f"\nCreating transformer datasets with target columns:")
    logger.info(f"Volatility target: {vol_target_col}")
    logger.info(f"Returns target: {ret_target_col}")
    logger.info(f"Using {'lagged' if use_lagged_targets else 'non-lagged'} targets")

    for split, df in datasets.items():
        logger.info(f"\nCreating dataset for {split} split:")
        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"Number of unique dates: {df['date'].nunique()}")
        logger.info(f"Number of unique symbols: {df['symbol'].nunique()}")
        
        # Validate data sorting and continuity
        logger.info("\nValidating data continuity:")
        grouped = df.groupby(['symbol', 'date'])
        sequence_stats = []
        
        for (symbol, date), group in grouped:
            group_size = len(group)
            sequence_stats.append({
                'symbol': symbol,
                'date': date,
                'size': group_size,
                'min_minute': group['minute'].min(),
                'max_minute': group['minute'].max(),
                'potential_sequences': max(0, group_size - sequence_length)
            })
        
        stats_df = pd.DataFrame(sequence_stats)
        logger.info(f"Group statistics:")
        logger.info(f"Total groups: {len(stats_df)}")
        logger.info(f"Groups with enough data (>={sequence_length}): {(stats_df['size'] >= sequence_length).sum()}")
        logger.info(f"Average group size: {stats_df['size'].mean():.1f}")
        logger.info(f"Min group size: {stats_df['size'].min()}")
        logger.info(f"Max group size: {stats_df['size'].max()}")
        logger.info(f"Total potential sequences: {stats_df['potential_sequences'].sum()}")
        
        # Handle missing values if needed
        nan_features = df[feature_cols].isna().sum()
        if nan_features.any():
            if model_type == 'xgboost':
                logger.info("Skipping NaN handling for XGBoost model")
            else:
                logger.info("\nHandling NaN values in features using handle_missing_values function...")
                df = handle_missing_values(df, feature_cols, model_type)
                
                # Verify no NaNs remain
                remaining_nans = df[feature_cols].isna().sum()
                if remaining_nans.any():
                    logger.error("Still found NaN values after handling:")
                    for col in feature_cols:
                        if remaining_nans[col] > 0:
                            logger.error(f"- {col}: {remaining_nans[col]} NaN values")
                    raise ValueError(f"Failed to handle all NaN values for {split} split")
                else:
                    logger.info("Successfully handled all NaN values")
        
        # Check for NaN values in targets
        nan_targets = df[[vol_target_col, ret_target_col]].isna().sum()
        if nan_targets.any():
            logger.error("\nFound NaN values in targets:")
            for col in [vol_target_col, ret_target_col]:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.error(f"- {col}: {nan_count} NaN values")
            raise ValueError(f"Found NaN values in targets for {split} split")
        
        # Create dataset with detailed error handling
        try:
            logger.info(f"\nCreating dataset with parameters:")
            logger.info(f"sequence_length: {sequence_length}")
            logger.info(f"sample_every: {sample_every}")
            logger.info(f"Number of features: {len(feature_cols)}")
            
            dataset = ThreeHeadedTransformerDataset(
                df=df,
                sequence_length=sequence_length,
                sample_every=sample_every,
                feature_cols=feature_cols,
                vol_target_col=vol_target_col,
                ret_target_col=ret_target_col,
                use_lagged_targets=use_lagged_targets
            )
            logger.info(f"Created dataset with {len(dataset)} sequences")
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            logger.error("\nDataFrame sample:")
            logger.error(df.head())
            logger.error("\nDataFrame info:")
            logger.error(df.info())
            raise
        
        # Create DataLoader
        shuffle = (split == 'train')
        transformer_loaders[split] = DataLoader(
            dataset,
            batch_size=2048,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        logger.info(f"Created DataLoader for {split} split")
        
        # Validate first batch
        try:
            first_batch = next(iter(transformer_loaders[split]))
            inputs, targets_dict = first_batch
            logger.info(f"First batch shapes:")
            logger.info(f"- inputs: {inputs.shape}")
            logger.info(f"- volatility targets: {targets_dict['volatility'].shape}")
            logger.info(f"- returns targets: {targets_dict['returns'].shape}")
        except Exception as e:
            logger.error(f"Error validating first batch: {str(e)}")
            raise
    
    return transformer_loaders

def load_parquet_chunk(file_info: Tuple[str, str, s3fs.S3FileSystem], selected_cols: List[str], subsample_fraction: Optional[float] = None) -> pd.DataFrame:
    """
    Load a single parquet file with optimized settings and optional subsampling.
    
    Parameters:
    -----------
    file_info : Tuple[str, str, s3fs.S3FileSystem]
        Tuple containing (file_path, date_str, filesystem)
    selected_cols : List[str]
        List of columns to load from the parquet file
    subsample_fraction : Optional[float]
        If provided, randomly sample this fraction of symbols from the file
    
    Returns:
    --------
    pd.DataFrame
        Loaded and preprocessed dataframe
    """
    file_path, date_str, fs = file_info
    
    if subsample_fraction is not None:
        # First, read just the 'symbol' column to get unique symbols
        with fs.open(file_path) as f:
            symbol_table = pq.read_table(f, columns=['symbol'])
            unique_symbols = pd.Series(symbol_table['symbol'].unique())
            
            # Randomly sample symbols
            n_symbols = len(unique_symbols)
            n_sample = max(1, int(n_symbols * subsample_fraction))
            sampled_symbols = unique_symbols.sample(n=n_sample)
            
            # Read full data only for sampled symbols
            table = pq.read_table(
                f,
                columns=selected_cols,
                filters=[('symbol', 'in', sampled_symbols.tolist())],
                memory_map=True,
                use_threads=True
            )
    else:
        # Read all data if no subsampling
        with fs.open(file_path) as f:
            table = pq.read_table(
                f,
                columns=selected_cols,
                memory_map=True,
                use_threads=True
            )
    
    df = table.to_pandas()
    df['date'] = pd.to_datetime(date_str)
    return df

def load_parquet_data(
    mode: str = "full",
    include_test: bool = False,
    test_n: int = 10,
    sequence_mode: bool = False,
    sequence_params: Dict = None,
    max_workers: int = 4,
    chunk_size: int = 10,
    debug: bool = False,
    model_type: str = None,
    subsample_fraction: Optional[float] = None,
    use_lagged_targets: bool = False
) -> Union[Tuple[Dict[str, pd.DataFrame], Tuple[StandardScaler, StandardScaler]], 
          Tuple[Dict[str, DataLoader], Tuple[StandardScaler, StandardScaler]]]:
    """
    Load and preprocess parquet data from S3 with parallel processing and intelligent subsampling.
    
    Parameters:
    -----------
    mode : str
        Either "full" or "subset"
    include_test : bool
        If True, include test set in returned datasets
    test_n : int
        Size of subset to use (10 or 100). Only used when mode="subset"
    sequence_mode : bool
        If True, return sequence DataLoaders instead of DataFrames
    sequence_params : Dict
        Parameters for sequence dataset creation (if sequence_mode=True)
    max_workers : int
        Number of parallel workers for file loading
    chunk_size : int
        Number of files to process in each chunk
    debug : bool
        If True, use TEST_SPLITS regardless of mode. Otherwise use SPLITS.
    model_type : str
        Type of model being trained ('xgboost', 'mlp', 'seq_mlp', 'naive')
    subsample_fraction : Optional[float]
        If provided, randomly subsample this fraction of data (0.0 to 1.0).
        Cannot be used with sequence_mode=True.
    use_lagged_targets : bool
        Whether to use lagged targets (default: False)
    Returns:
    --------
    Union[Tuple[Dict[str, pd.DataFrame], Tuple[StandardScaler, StandardScaler]], 
          Tuple[Dict[str, DataLoader], Tuple[StandardScaler, StandardScaler]]]
        Either (DataFrames, (feature_scaler, target_scaler)) or (DataLoaders, (feature_scaler, target_scaler)) depending on sequence_mode
    """
    logger = logging.getLogger('model_training')
    logger.info(f"Loading parquet data with subsample_fraction: {subsample_fraction}")
    
    # Validate subsample_fraction
    if subsample_fraction is not None:
        # Allow subsampling for transformer but not other sequence models
        if sequence_mode and model_type not in ['transformer']:
            raise ValueError("subsample_fraction cannot be used with this sequence model type")
        if not 0.0 < subsample_fraction <= 1.0:
            raise ValueError("subsample_fraction must be between 0.0 and 1.0")
        logger.info(f"Will subsample {subsample_fraction:.1%} of the data")
    
    dataset_mode = "FULL" if mode == "full" else f"SUBSET_TOP_{test_n}"
    logger.info(f"\nLoading parquet data ({dataset_mode}, include_test={include_test}, "
                f"sequence_mode={sequence_mode})")
    start_time = time.time()
    
    # Use TEST_SPLITS only when in debug mode
    splits = TEST_SPLITS if debug else SPLITS
    logger.info(f"Using {'TEST_SPLITS' if debug else 'SPLITS'} for date ranges")
    
    # Initialize S3 filesystem
    logger.info("Initializing S3 filesystem...")
    fs = s3fs.S3FileSystem(anon=False)
    
    # Get NYSE calendar
    logger.info("Getting NYSE calendar...")
    nyse = mcal.get_calendar('NYSE')
    
    # Get needed dates
    logger.info("Calculating needed trading days...")
    needed_dates = set()
    for split, (start_date, end_date) in splits.items():
        if split == 'test' and not include_test:
            continue
        date_range = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        trading_days = schedule.index.strftime('%Y%m%d')
        needed_dates.update(trading_days)
    
    logger.info(f"Need to load {len(needed_dates)} trading days between {min(needed_dates)} and {max(needed_dates)}")
    
    # Prepare file information
    logger.info("Preparing file information...")
    base_path = f"s3://{BUCKET_NAME}/data/features/attention_df/"
    input_path = base_path + ("all/" if mode == "full" else "top_n/")
    
    file_infos = []
    seen_dates = set()
    for date in sorted(needed_dates):
        if date in seen_dates:
            continue
        pattern = f"{date}.parquet" if mode == "full" else f"{date}.top{test_n}.parquet"
        file_path = input_path + pattern
        if fs.exists(file_path.replace("s3://", "")):
            file_infos.append((file_path.replace("s3://", ""), date, fs))
            seen_dates.add(date)
            logger.info(f"Found file for date {date}")
        else:
            logger.warning(f"Missing file for trading day {date}")
    
    if not file_infos:
        raise ValueError("No parquet files found for the specified date range")
    
    logger.info(f"Found {len(file_infos)} files to process")
    
    # Define required columns
    vol_target = "Y_log_vol_10min_lag_1m" if use_lagged_targets else "Y_log_vol_10min"
    ret_target = "Y_log_ret_60min_lag_1m" if use_lagged_targets else "Y_log_ret_60min"
    required_cols = selected_features + ["symbol", "minute", vol_target, ret_target]
    
    # Initialize empty datasets dictionary
    datasets = {split: pd.DataFrame() for split in splits if split != 'test' or include_test}
    
    # Process each split separately
    for split, (start_date, end_date) in splits.items():
        if split == 'test' and not include_test:
            continue
            
        logger.info(f"\nProcessing {split} split ({start_date} to {end_date})...")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter file_infos for current split
        split_files = [
            fi for fi in file_infos 
            if start_dt <= pd.to_datetime(fi[1]) <= end_dt
        ]
        logger.info(f"Found {len(split_files)} files for {split} split")
        
        split_dfs = []
        # Process files in chunks
        for i in range(0, len(split_files), chunk_size):
            chunk = split_files[i:i + chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1} of {(len(split_files) + chunk_size - 1)//chunk_size}")
            
            # Process chunk in parallel with subsampling
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        load_parquet_chunk, 
                        file_info, 
                        required_cols,
                        subsample_fraction  # Pass subsample_fraction to chunk loader
                    ): file_info 
                    for file_info in chunk
                }
                
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    try:
                        df = future.result()
                        logger.info(f"Loaded {len(df):,} rows for date {df['date'].iloc[0].date()}")
                        split_dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error loading {file_info[0]}: {str(e)}")
                        raise
            
            # Combine chunk results immediately
            if len(split_dfs) >= chunk_size:
                chunk_df = pd.concat(split_dfs, ignore_index=True)
                chunk_df = chunk_df.sort_values(['date', 'symbol', 'minute'])
                split_dfs = [chunk_df]
        
        # Combine all chunks for this split
        if split_dfs:
            logger.info(f"Combining {len(split_dfs)} dataframes for {split} split...")
            datasets[split] = pd.concat(split_dfs, ignore_index=True)
            datasets[split] = datasets[split].sort_values(['date', 'symbol', 'minute'])
            logger.info(f"Combined {split} split: {len(datasets[split]):,} rows")
        else:
            logger.warning(f"No data found for {split} split")

    # Preprocess features for each split
    logger.info("\nPreprocessing features...")
    feature_scaler = None
    for split in datasets.keys():
        if not datasets[split].empty:
            if split == 'train':
                # Fit scaler on training data (if needed)
                datasets[split], feature_scaler = preprocess_data(datasets[split], selected_features, model_type)
                logger.info(f"Fitted feature scaler on {split} split")
            else:
                # Only transform if a scaler exists (i.e. for non-XGBoost models)
                if feature_scaler is not None:
                    datasets[split][selected_features] = feature_scaler.transform(datasets[split][selected_features])
                    logger.info(f"Transformed {split} split using training scaler")
                else:
                    logger.info(f"Skipping scaling for {split} split (no scaler fitted)")

    # Convert DataFrames to appropriate dataset type based on model
    if sequence_mode:
        if model_type == "transformer":
            logger.info("\nCreating transformer datasets...")
            datasets = create_transformer_datasets(
                datasets,
                feature_cols=selected_features,
                sequence_length=sequence_params.get("sequence_length", 60),
                sample_every=sequence_params.get("sample_every", 10),
                vol_target_col=vol_target,
                ret_target_col=ret_target,
                model_type=model_type,
                use_lagged_targets=use_lagged_targets
            )
            
            # Fit scalers for transformer model
            train_loader = datasets['train']
            all_vol_targets = []
            all_ret_targets = []
            
            for batch_idx, (inputs, targets_dict) in enumerate(train_loader):
                all_vol_targets.append(targets_dict['volatility'].numpy())
                all_ret_targets.append(targets_dict['returns'].numpy())
                
                if batch_idx < 3:
                    logger.info(f"Batch {batch_idx} stats:")
                    logger.info(f"X - min: {inputs.min():.3f}, max: {inputs.max():.3f}, mean: {inputs.mean():.3f}")
                    logger.info(f"y_vol - min: {targets_dict['volatility'].min():.3f}, max: {targets_dict['volatility'].max():.3f}")
                    logger.info(f"y_ret - min: {targets_dict['returns'].min():.3f}, max: {targets_dict['returns'].max():.3f}")
            
            # Fit transformer scalers
            all_vol_targets = np.concatenate(all_vol_targets)
            all_ret_targets = np.concatenate(all_ret_targets)
            
            vol_scaler = StandardScaler()
            ret_scaler = StandardScaler()
            vol_scaler.fit(all_vol_targets.reshape(-1, 1))
            ret_scaler.fit(all_ret_targets.reshape(-1, 1))
            
            target_scalers = (vol_scaler, ret_scaler)
            
        elif model_type == "seq_mlp":
            logger.info("\nCreating sequence MLP datasets...")
            datasets = create_sequence_datasets(
                datasets,
                feature_cols=selected_features,
                short_seq_len=sequence_params.get("short_seq_len", 20),
                long_seq_len=sequence_params.get("long_seq_len", 60),
                sample_every=sequence_params.get("sample_every", 10),
                target_col=vol_target,
                use_lagged_targets=use_lagged_targets
            )
            
            # Fit scaler for sequence MLP model
            train_loader = datasets['train']
            all_targets = []
            
            for batch_idx, (_, targets) in enumerate(train_loader):
                all_targets.append(targets.numpy())
                
                if batch_idx < 3:
                    logger.info(f"Batch {batch_idx} stats:")
                    logger.info(f"y - min: {targets.min():.3f}, max: {targets.max():.3f}, mean: {targets.mean():.3f}")
            
            # Fit sequence MLP scaler
            all_targets = np.concatenate(all_targets)
            target_scaler = StandardScaler()
            target_scaler.fit(all_targets.reshape(-1, 1))
            
            target_scalers = (target_scaler, None)
            
        else:
            raise ValueError(f"Unknown sequence model type: {model_type}")
        
        logger.info("\nTarget scaling parameters:")
        if model_type == "transformer":
            logger.info(f"Volatility - mean: {vol_scaler.mean_[0]:.3f}, scale: {vol_scaler.scale_[0]:.3f}")
            logger.info(f"Returns - mean: {ret_scaler.mean_[0]:.3f}, scale: {ret_scaler.scale_[0]:.3f}")
        else:
            logger.info(f"Target - mean: {target_scaler.mean_[0]:.3f}, scale: {target_scaler.scale_[0]:.3f}")
    else:
        # Handle non-sequence cases
        target_scalers = (feature_scaler, None)
    
    return datasets, feature_scaler, target_scalers