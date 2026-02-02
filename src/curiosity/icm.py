"""
Intrinsic Curiosity Module (ICM) для автономного исследования торговых стратегий.

Реализует curiosity-driven exploration через forward/inverse dynamics модели
с enterprise patterns для масштабируемой системы исследования.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ICMConfig:
    """Конфигурация для Intrinsic Curiosity Module."""
    
    state_dim: int = 256
    action_dim: int = 10
    feature_dim: int = 128
    hidden_dim: int = 256
    
    # Веса потерь
    forward_loss_weight: float = 0.2
    inverse_loss_weight: float = 0.8
    curiosity_reward_weight: float = 1.0
    
    # Параметры обучения
    learning_rate: float = 1e-4
    batch_size: int = 256
    
    # Crypto-trading специфичные параметры
    market_features: int = 50  # Технические индикаторы
    portfolio_features: int = 20  # Состояние портфеля
    risk_features: int = 10  # Риск-метрики
    
    #  cloud-native settings
    distributed_training: bool = True
    checkpoint_interval: int = 1000
    metrics_enabled: bool = True


class FeatureEncoder(nn.Module):
    """
    Кодировщик состояний для ICM с поддержкой crypto-trading данных.
    
    Применяет design pattern "Feature Representation Learning"
    для эффективного представления состояний рынка.
    """
    
    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config
        
        # Многослойная архитектура для разных типов данных
        self.market_encoder = nn.Sequential(
            nn.Linear(config.market_features, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(config.portfolio_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.Dropout(0.1)
        )
        
        self.risk_encoder = nn.Sequential(
            nn.Linear(config.risk_features, config.hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 4)
        )
        
        # Объединяющий слой
        total_features = config.hidden_dim // 2 + config.hidden_dim // 2 + config.hidden_dim // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, config.feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.feature_dim)
        )
        
        logger.info(f"Feature encoder initialized with {total_features} input features")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Кодирование состояния в compact feature representation.
        
        Args:
            state: Tensor формы [batch_size, state_dim]
            
        Returns:
            Encoded features: [batch_size, feature_dim]
        """
        batch_size = state.size(0)
        
        # Разделение входного состояния на компоненты
        market_data = state[:, :self.config.market_features]
        portfolio_data = state[:, 
            self.config.market_features:self.config.market_features + self.config.portfolio_features
        ]
        risk_data = state[:, -self.config.risk_features:]
        
        # Кодирование каждого компонента
        market_features = self.market_encoder(market_data)
        portfolio_features = self.portfolio_encoder(portfolio_data)
        risk_features = self.risk_encoder(risk_data)
        
        # Объединение и финальное кодирование
        combined_features = torch.cat([market_features, portfolio_features, risk_features], dim=1)
        encoded_state = self.fusion_layer(combined_features)
        
        return encoded_state


class ForwardModel(nn.Module):
    """
    Forward Dynamics Model для предсказания следующего состояния.
    
    Использует design pattern "Predictive Modeling" для точного
    прогнозирования динамики рынка.
    """
    
    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config
        
        # Архитектура с residual connections для стабильного обучения
        self.action_encoder = nn.Linear(config.action_dim, config.hidden_dim // 4)
        
        self.forward_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.feature_dim + config.hidden_dim // 4, config.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(config.hidden_dim),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(config.hidden_dim),
                nn.Dropout(0.1)
            ),
            nn.Linear(config.hidden_dim, config.feature_dim)
        ])
        
        logger.info("Forward model initialized for next state prediction")
    
    def forward(self, state_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Предсказание следующего состояния по текущему состоянию и действию.
        
        Args:
            state_features: Encoded state features [batch_size, feature_dim]
            action: Action vector [batch_size, action_dim]
            
        Returns:
            Predicted next state features [batch_size, feature_dim]
        """
        # Кодирование действия
        action_encoded = F.relu(self.action_encoder(action))
        
        # Объединение состояния и действия
        state_action = torch.cat([state_features, action_encoded], dim=1)
        
        # Прохождение через forward network с residual connection
        x = state_action
        for i, layer in enumerate(self.forward_net[:-1]):
            residual = x if i > 0 else None
            x = layer(x)
            if residual is not None and residual.shape == x.shape:
                x = x + residual
        
        # Финальный слой без residual
        predicted_next_state = self.forward_net[-1](x)
        
        return predicted_next_state


class InverseModel(nn.Module):
    """
    Inverse Dynamics Model для предсказания действия между состояниями.
    
    Применяет design pattern "Action Understanding" для изучения
    контролируемых аспектов окружения.
    """
    
    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config
        
        # Симметричная архитектура для обработки пары состояний
        self.inverse_net = nn.Sequential(
            nn.Linear(config.feature_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(config.hidden_dim // 2, config.action_dim)
        )
        
        logger.info("Inverse model initialized for action prediction")
    
    def forward(self, state_features: torch.Tensor, next_state_features: torch.Tensor) -> torch.Tensor:
        """
        Предсказание действия между двумя состояниями.
        
        Args:
            state_features: Current state features [batch_size, feature_dim]
            next_state_features: Next state features [batch_size, feature_dim]
            
        Returns:
            Predicted action [batch_size, action_dim]
        """
        # Объединение состояний
        state_pair = torch.cat([state_features, next_state_features], dim=1)
        
        # Предсказание действия
        predicted_action = self.inverse_net(state_pair)
        
        return predicted_action


class CuriosityRewardCalculator:
    """
    Вычислитель intrinsic reward на основе prediction error.
    
    Использует design pattern "Reward Engineering" для формирования
    эффективных сигналов любопытства.
    """
    
    def __init__(self, config: ICMConfig):
        self.config = config
        self.prediction_errors = []
        self.running_mean = 0.0
        self.running_var = 1.0
        self.alpha = 0.01  # Коэффициент для экспоненциального сглаживания
        
        logger.info("Curiosity reward calculator initialized")
    
    def calculate_curiosity_reward(
        self,
        predicted_next_state: torch.Tensor,
        actual_next_state: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Вычисление curiosity reward на основе prediction error.
        
        Args:
            predicted_next_state: Предсказанное следующее состояние
            actual_next_state: Реальное следующее состояние
            normalize: Применить нормализацию
            
        Returns:
            Curiosity rewards для каждого sample в batch
        """
        # Вычисление L2 prediction error
        prediction_error = F.mse_loss(
            predicted_next_state, 
            actual_next_state, 
            reduction='none'
        ).mean(dim=1)
        
        if normalize:
            # Обновление running statistics
            current_mean = prediction_error.mean().item()
            current_var = prediction_error.var().item()
            
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * current_mean
            self.running_var = (1 - self.alpha) * self.running_var + self.alpha * current_var
            
            # Нормализация reward
            std = (self.running_var + 1e-8) ** 0.5
            normalized_error = (prediction_error - self.running_mean) / std
            curiosity_reward = torch.clamp(normalized_error, 0, 5)  # Ограничение сверху
        else:
            curiosity_reward = prediction_error
        
        # Сохранение для статистики
        self.prediction_errors.extend(prediction_error.detach().cpu().numpy())
        if len(self.prediction_errors) > 10000:
            self.prediction_errors = self.prediction_errors[-5000:]
        
        return curiosity_reward * self.config.curiosity_reward_weight


class ICMTrainer:
    """
    Тренер для Intrinsic Curiosity Module с advanced optimization.
    
    Реализует design pattern "Distributed Learning" для
    эффективного обучения на больших объемах данных.
    """
    
    def __init__(self, config: ICMConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Инициализация моделей
        self.feature_encoder = FeatureEncoder(config).to(device)
        self.forward_model = ForwardModel(config).to(device)
        self.inverse_model = InverseModel(config).to(device)
        self.curiosity_calculator = CuriosityRewardCalculator(config)
        
        # Оптимизаторы с разными learning rates
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_encoder.parameters(), 'lr': config.learning_rate},
            {'params': self.forward_model.parameters(), 'lr': config.learning_rate * 0.5},
            {'params': self.inverse_model.parameters(), 'lr': config.learning_rate * 1.5}
        ], weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        self.training_step = 0
        self.metrics = {
            'forward_loss': [],
            'inverse_loss': [],
            'total_loss': [],
            'curiosity_rewards': []
        }
        
        logger.info(f"ICM trainer initialized on device: {device}")
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Выполнение одного шага обучения ICM.
        
        Args:
            states: Batch of current states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            next_states: Batch of next states [batch_size, state_dim]
            
        Returns:
            Dictionary с метриками обучения
        """
        self.optimizer.zero_grad()
        
        # Кодирование состояний
        state_features = self.feature_encoder(states)
        next_state_features = self.feature_encoder(next_states)
        
        # Forward model prediction
        predicted_next_features = self.forward_model(state_features, actions)
        
        # Inverse model prediction
        predicted_actions = self.inverse_model(state_features, next_state_features)
        
        # Вычисление потерь
        forward_loss = F.mse_loss(predicted_next_features, next_state_features.detach())
        inverse_loss = F.mse_loss(predicted_actions, actions)
        
        # Общая потеря
        total_loss = (
            self.config.forward_loss_weight * forward_loss +
            self.config.inverse_loss_weight * inverse_loss
        )
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.feature_encoder.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.inverse_model.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()
        
        # Вычисление curiosity reward
        with torch.no_grad():
            curiosity_rewards = self.curiosity_calculator.calculate_curiosity_reward(
                predicted_next_features, next_state_features
            )
        
        # Обновление метрик
        metrics = {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_loss': total_loss.item(),
            'curiosity_reward_mean': curiosity_rewards.mean().item(),
            'curiosity_reward_std': curiosity_rewards.std().item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        for key, value in metrics.items():
            if key != 'learning_rate':
                self.metrics[key.replace('_mean', '').replace('_std', '')].append(value)
        
        self.training_step += 1
        
        return metrics
    
    def get_curiosity_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Получение curiosity reward для evaluation.
        
        Args:
            state: Current state
            action: Executed action
            next_state: Resulting state
            
        Returns:
            Curiosity reward
        """
        with torch.no_grad():
            state_features = self.feature_encoder(state)
            next_state_features = self.feature_encoder(next_state)
            predicted_next_features = self.forward_model(state_features, action)
            
            curiosity_reward = self.curiosity_calculator.calculate_curiosity_reward(
                predicted_next_features, next_state_features
            )
            
        return curiosity_reward
    
    def save_checkpoint(self, filepath: str) -> None:
        """Сохранение checkpoint модели."""
        checkpoint = {
            'feature_encoder': self.feature_encoder.state_dict(),
            'forward_model': self.forward_model.state_dict(),
            'inverse_model': self.inverse_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'metrics': self.metrics
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Загрузка checkpoint модели."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.feature_encoder.load_state_dict(checkpoint['feature_encoder'])
        self.forward_model.load_state_dict(checkpoint['forward_model'])
        self.inverse_model.load_state_dict(checkpoint['inverse_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_step = checkpoint['training_step']
        self.metrics = checkpoint['metrics']
        
        logger.info(f"Checkpoint loaded from {filepath}")


class CryptoICMEnvironment:
    """
    Специализированная обертка для интеграции ICM с crypto trading environment.
    
    Применяет design pattern "Environment Adaptation" для
    seamless интеграции с торговыми системами.
    """
    
    def __init__(self, base_env, icm_trainer: ICMTrainer, reward_mix: float = 0.1):
        self.base_env = base_env
        self.icm_trainer = icm_trainer
        self.reward_mix = reward_mix  # Доля intrinsic reward в общем reward
        
        self.last_state = None
        self.last_action = None
        
        logger.info(f"Crypto ICM environment initialized with reward mix: {reward_mix}")
    
    def step(self, action):
        """
        Выполнение шага с добавлением curiosity reward.
        
        Args:
            action: Действие агента
            
        Returns:
            Tuple (next_state, total_reward, done, info)
        """
        # Выполнение действия в базовой среде
        next_state, extrinsic_reward, done, info = self.base_env.step(action)
        
        # Вычисление curiosity reward если есть предыдущее состояние
        intrinsic_reward = 0.0
        if self.last_state is not None and self.last_action is not None:
            state_tensor = torch.FloatTensor(self.last_state).unsqueeze(0).to(self.icm_trainer.device)
            action_tensor = torch.FloatTensor(self.last_action).unsqueeze(0).to(self.icm_trainer.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.icm_trainer.device)
            
            curiosity_reward = self.icm_trainer.get_curiosity_reward(
                state_tensor, action_tensor, next_state_tensor
            )
            intrinsic_reward = curiosity_reward.item()
        
        # Объединение extrinsic и intrinsic rewards
        total_reward = extrinsic_reward + self.reward_mix * intrinsic_reward
        
        # Обновление состояния для следующего шага
        self.last_state = next_state.copy() if hasattr(next_state, 'copy') else next_state
        self.last_action = action.copy() if hasattr(action, 'copy') else action
        
        # Добавление curiosity info
        info['intrinsic_reward'] = intrinsic_reward
        info['extrinsic_reward'] = extrinsic_reward
        info['reward_mix'] = self.reward_mix
        
        return next_state, total_reward, done, info
    
    def reset(self):
        """Сброс среды."""
        state = self.base_env.reset()
        self.last_state = state.copy() if hasattr(state, 'copy') else state
        self.last_action = None
        return state


def create_icm_system(config: ICMConfig) -> Tuple[ICMTrainer, CryptoICMEnvironment]:
    """
    Factory function для создания complete ICM system.
    
    Args:
        config: Конфигурация ICM
        
    Returns:
        Tuple (ICM trainer, ICM-wrapped environment)
    """
    # Инициализация ICM trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    icm_trainer = ICMTrainer(config, device)
    
    logger.info("ICM system created successfully")
    logger.info(f"Feature encoder parameters: {sum(p.numel() for p in icm_trainer.feature_encoder.parameters())}")
    logger.info(f"Forward model parameters: {sum(p.numel() for p in icm_trainer.forward_model.parameters())}")
    logger.info(f"Inverse model parameters: {sum(p.numel() for p in icm_trainer.inverse_model.parameters())}")
    
    return icm_trainer


if __name__ == "__main__":
    # Пример использования ICM для crypto trading
    config = ICMConfig(
        state_dim=80,  # 50 market + 20 portfolio + 10 risk
        action_dim=5,  # Buy/Sell/Hold для разных активов
        feature_dim=64,
        hidden_dim=128
    )
    
    icm_trainer = create_icm_system(config)
    
    # Создание synthetic данных для демонстрации
    batch_size = 32
    states = torch.randn(batch_size, config.state_dim)
    actions = torch.randn(batch_size, config.action_dim)
    next_states = torch.randn(batch_size, config.state_dim)
    
    # Обучение ICM
    metrics = icm_trainer.train_step(states, actions, next_states)
    print("Training metrics:", metrics)
    
    # Получение curiosity reward
    curiosity_reward = icm_trainer.get_curiosity_reward(
        states[:1], actions[:1], next_states[:1]
    )
    print(f"Curiosity reward: {curiosity_reward.item():.4f}")