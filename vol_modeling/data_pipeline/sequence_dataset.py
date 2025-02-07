import torch
from torch.utils.data import Dataset
import numpy as np
import logging

class SequenceVolatilityDataset(Dataset):
    def __init__(
        self, 
        df, 
        short_seq_len=20, 
        long_seq_len=60,   # For example
        sample_every=10,   # Jump size for sampling
        feature_cols=None,
        target_col="Y_log_vol_10min",  # Default to non-lagged
        use_lagged_targets=False  # Default to non-lagged
    ):
        """
        Create sequence dataset with proper sample counting to avoid uninitialized data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        short_seq_len : int
            Length of short sequence window
        long_seq_len : int
            Length of long sequence window
        sample_every : int
            Sampling interval
        feature_cols : List[str]
            Feature columns to use
        target_col : str
            Target column name (default: Y_log_vol_10min)
        use_lagged_targets : bool
            Whether to use lagged targets (default: False)
        """
        assert not (use_lagged_targets and not "lag" in target_col)
        # First pass: count total valid samples properly
        grouped = df.groupby(["symbol", "date"], group_keys=False)
        total_samples = 0
        
        for (sym, d), group_df in grouped:
            group_df = group_df.sort_values("minute")
            max_idx = len(group_df)
            
            # Skip groups that are too short
            if max_idx <= short_seq_len:
                continue
                
            # Count exactly how many valid sequences this group will generate
            valid_sequences = 0
            for idx in range(short_seq_len, max_idx - 1, sample_every):
                end_idx = idx + 1
                
                # Verify sequence lengths
                short_valid = (end_idx - short_seq_len) >= 0
                long_valid = not long_seq_len or (end_idx - long_seq_len) >= 0
                
                if short_valid and long_valid:
                    valid_sequences += 1
            
            total_samples += valid_sequences
        
        # Pre-allocate tensors with exact size
        num_features = len(feature_cols)
        self.X_short = torch.empty((total_samples, short_seq_len, num_features), dtype=torch.float32)
        if long_seq_len:
            self.X_long = torch.empty((total_samples, long_seq_len, num_features), dtype=torch.float32)
        else:
            self.X_long = None
        self.y = torch.empty(total_samples, dtype=torch.float32)
        
        # Second pass: fill tensors
        sample_idx = 0
        for (sym, d), group_df in grouped:
            group_df = group_df.sort_values("minute")
            arr = group_df[feature_cols].values
            targets = group_df[target_col].values
            
            max_idx = arr.shape[0]
            if max_idx <= short_seq_len:
                continue

            for idx in range(short_seq_len, max_idx - 1, sample_every):
                end_idx = idx + 1
                X_short = arr[end_idx - short_seq_len : end_idx]
                if long_seq_len:
                    X_long = arr[end_idx - long_seq_len : end_idx]

                # Double-check sequence lengths
                if len(X_short) != short_seq_len:
                    continue
                if long_seq_len and len(X_long) != long_seq_len:
                    continue

                self.X_short[sample_idx] = torch.tensor(X_short)
                if self.X_long is not None:
                    self.X_long[sample_idx] = torch.tensor(X_long)
                self.y[sample_idx] = targets[idx]
                sample_idx += 1
        
        assert sample_idx == total_samples, f"Mismatch in sample count: allocated {total_samples} but filled {sample_idx}"

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        if self.X_long is not None:
            return self.X_short[i], self.X_long[i], self.y[i]
        return self.X_short[i], None, self.y[i]


class ThreeHeadedTransformerDataset(Dataset):
    def __init__(
        self,
        df,
        sequence_length=30,
        sample_every=10,
        feature_cols=None,
        vol_target_col="Y_log_vol_10min",  # Default to non-lagged
        ret_target_col="Y_log_ret_60min",  # Default to non-lagged
        use_lagged_targets=False  # Default to non-lagged
    ):
        """
        Create a dataset that generates one sequence per sample along with two target values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        sequence_length : int
            Length of sequences to create
        sample_every : int
            Sampling interval
        feature_cols : List[str]
            Feature columns to use
        vol_target_col : str
            Volatility target column name (default: Y_log_vol_10min)
        ret_target_col : str
            Returns target column name (default: Y_log_ret_60min)
        use_lagged_targets : bool
            Whether to use lagged targets (default: False)
        """
        # Validate target columns match lagged/non-lagged setting
        if use_lagged_targets:
            if "lag" not in vol_target_col or "lag" not in ret_target_col:
                raise ValueError(f"Using lagged targets but target columns {vol_target_col}, {ret_target_col} do not both contain 'lag'")
        else:
            if "lag" in vol_target_col or "lag" in ret_target_col:
                raise ValueError(f"Not using lagged targets but target columns {vol_target_col}, {ret_target_col} contain 'lag'")
        
        logger = logging.getLogger('model_training')
        logger.setLevel(logging.DEBUG)  # Ensure level is set low enough
        
        # If no handlers are attached, add a StreamHandler
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        # First pass: count total valid sequences
        logger.info("\nCounting valid sequences:")
        
        # Log variance statistics for targets
        logger.info("\nTarget Variance Statistics:")
        logger.info(f"Volatility target ({vol_target_col}):")
        logger.info(f"Mean: {df[vol_target_col].mean():.4f}")
        logger.info(f"Std: {df[vol_target_col].std():.4f}")
        logger.info(f"Min: {df[vol_target_col].min():.4f}")
        logger.info(f"Max: {df[vol_target_col].max():.4f}")
        #assert that "date" is present in df
        assert "date" in df.columns, "date column is required"
        # Group data by symbol and date to ensure sequential consistency
        grouped = df.groupby(["symbol", "date"], group_keys=False)
        total_samples = 0
        
        # First pass: count total valid sequences
        group_stats = []
        
        for (sym, d), group_df in grouped:
            group_df = group_df.sort_values("minute")
            max_idx = len(group_df)
            
            if max_idx <= sequence_length:
                logger.info(f"Skipping group ({sym}, {d}): length {max_idx} <= sequence_length {sequence_length}")
                continue
            
            num_sequences = len(range(sequence_length, max_idx - 1, sample_every))
            total_samples += num_sequences
            
            group_stats.append({
                'symbol': sym,
                'date': d,
                'length': max_idx,
                'sequences': num_sequences
            })
        
        logger.info(f"Total valid samples counted: {total_samples}")
        if group_stats:
            lengths = [s['length'] for s in group_stats]
            sequences = [s['sequences'] for s in group_stats]
            logger.info(f"Group statistics:")
            logger.info(f"- Number of valid groups: {len(group_stats)}")
            logger.info(f"- Average group length: {sum(lengths)/len(lengths):.1f}")
            logger.info(f"- Min group length: {min(lengths)}")
            logger.info(f"- Max group length: {max(lengths)}")
            logger.info(f"- Average sequences per group: {sum(sequences)/len(sequences):.1f}")

        # Pre-allocate tensors for sequences and targets
        num_features = len(feature_cols)
        self.X = torch.empty((total_samples, sequence_length, num_features), dtype=torch.float32)
        self.vol_y = torch.empty(total_samples, dtype=torch.float32)
        self.ret_y = torch.empty(total_samples, dtype=torch.float32)

        # Second pass: extract and fill tensors
        sample_idx = 0
        logger.info("\nFilling sequences:")
        
        for (sym, d), group_df in grouped:
            group_df = group_df.sort_values("minute")
            arr = group_df[feature_cols].values.astype(np.float32)
            vol_targets = group_df[vol_target_col].values.astype(np.float32)
            ret_targets = group_df[ret_target_col].values.astype(np.float32)
            
            max_idx = arr.shape[0]
            if max_idx <= sequence_length:
                continue

            sequences_filled = 0
            for idx in range(sequence_length, max_idx - 1, sample_every):
                end_idx = idx + 1
                X_seq = arr[end_idx - sequence_length : end_idx]
                
                # Double-check sequence length
                if len(X_seq) != sequence_length:
                    logger.warning(f"Unexpected sequence length: {len(X_seq)} != {sequence_length}")
                    continue
                
                # Extract targets at the specified offsets
                vol_target = vol_targets[idx]
                ret_target = ret_targets[idx]

                self.X[sample_idx] = torch.tensor(X_seq)
                self.vol_y[sample_idx] = torch.tensor(vol_target)
                self.ret_y[sample_idx] = torch.tensor(ret_target)
                sample_idx += 1
                sequences_filled += 1
            
            if sequences_filled > 0:
                logger.info(f"Filled {sequences_filled} sequences for ({sym}, {d})")

        logger.info(f"\nSequence creation summary:")
        logger.info(f"Total samples allocated: {total_samples}")
        logger.info(f"Total samples filled: {sample_idx}")
        
        assert sample_idx == total_samples, (
            f"Mismatch in sample count: allocated {total_samples} but filled {sample_idx}"
        )

    def __len__(self):
        return len(self.vol_y)

    def __getitem__(self, idx):
        input_seq = self.X[idx]
        target_dict = {
            'volatility': self.vol_y[idx],
            'returns': self.ret_y[idx]
        }
        return input_seq, target_dict
