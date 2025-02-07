"""Neural network models"""
class MLPModel(VolatilityModel):
    """Simple feedforward network"""
    def build_network(self, input_dim: int) -> nn.Module:
        pass

class LSTMModel(VolatilityModel):
    """LSTM-based model"""
    def build_network(self, input_dim: int, 
                     hidden_dim: int) -> nn.Module:
        pass
