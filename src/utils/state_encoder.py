"""
State Encoder utilities для curiosity-driven exploration.

Реализует advanced state encoding methods с Context7 enterprise patterns
для efficient state representation в crypto trading environments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StateEncoderConfig:
    """Конфигурация для state encoder."""
    
    input_dim: int = 256
    encoding_dim: int = 64
    hidden_dims: List[int] = None
    
    # Encoding strategies
    normalization: bool = True
    feature_selection: bool = True
    temporal_encoding: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class CryptoStateEncoder(nn.Module):
    """
    Advanced state encoder для crypto trading states.
    
    Применяет Context7 паттерн "Feature Engineering" для
    optimal state representation.
    """
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Multi-component encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(config.input_dim // 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1)
        )
        
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(config.input_dim // 4, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )
        
        self.risk_encoder = nn.Sequential(
            nn.Linear(config.input_dim // 4, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # Fusion layer
        total_features = 128 + 64 + 32
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, config.encoding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.encoding_dim)
        )
        
        logger.info(f"Crypto state encoder initialized: {config.input_dim} -> {config.encoding_dim}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state в compact representation."""
        batch_size = state.size(0)
        
        # Split state по компонентам
        market_data = state[:, :self.config.input_dim // 2]
        portfolio_data = state[:, self.config.input_dim // 2:3 * self.config.input_dim // 4]
        risk_data = state[:, 3 * self.config.input_dim // 4:]
        
        # Encode каждый компонент
        market_features = self.market_encoder(market_data)
        portfolio_features = self.portfolio_encoder(portfolio_data)
        risk_features = self.risk_encoder(risk_data)
        
        # Fusion
        combined = torch.cat([market_features, portfolio_features, risk_features], dim=1)
        encoded = self.fusion_layer(combined)
        
        return encoded


def create_state_encoder(config: StateEncoderConfig) -> CryptoStateEncoder:
    """Factory function для создания state encoder."""
    return CryptoStateEncoder(config)


if __name__ == "__main__":
    config = StateEncoderConfig(input_dim=256, encoding_dim=64)
    encoder = create_state_encoder(config)
    
    # Test encoding
    batch_size = 32
    test_states = torch.randn(batch_size, 256)
    
    with torch.no_grad():
        encoded_states = encoder(test_states)
    
    print(f"Input shape: {test_states.shape}")
    print(f"Encoded shape: {encoded_states.shape}")
    print(f"Encoding dimension: {encoded_states.size(1)}")