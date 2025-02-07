"""Dataset and DataLoader implementations"""
class VolatilityDataset(Dataset):
    def __init__(self, features: np.ndarray, 
                 labels: np.ndarray):
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

# evaluation/
