"""
Never Give Up (NGU) Agent для persistent exploration в crypto trading.

Реализует state-of-the-art exploration through episodic memory и RND
с Context7 enterprise patterns для scalable long-term exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import faiss
import time
from abc import ABC, abstractmethod

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NGUConfig:
    """Конфигурация для Never Give Up agent."""
    
    # Основные параметры
    state_dim: int = 256
    action_dim: int = 10
    embedding_dim: int = 64
    
    # Episodic memory parameters
    episodic_memory_capacity: int = 50000
    num_neighbors: int = 10
    episodic_bonus_constant: float = 0.0001
    episodic_bonus_maximum: float = 5.0
    
    # RND parameters
    rnd_network_dim: int = 256
    rnd_learning_rate: float = 1e-4
    rnd_update_frequency: int = 4
    
    # Exploration schedule
    exploration_beta_schedule: List[float] = field(default_factory=lambda: [0.3, 0.1, 0.01])
    exploration_gamma_schedule: List[float] = field(default_factory=lambda: [0.99, 0.995, 0.999])
    
    # Crypto trading specific
    market_regime_memory: bool = True
    portfolio_state_memory: bool = True
    risk_state_memory: bool = True
    temporal_context_length: int = 20
    
    # Context7 enterprise settings
    distributed_memory: bool = True
    memory_compression: bool = True
    faiss_gpu: bool = True
    checkpoint_interval: int = 5000
    
    # Advanced parameters
    memory_replacement_strategy: str = "fifo"  # "fifo", "lru", "random"
    similarity_threshold: float = 0.1
    bonus_normalization: bool = True
    state_normalization: bool = True


class EpisodicMemory:
    """
    Episodic memory для NGU с efficient similarity search.
    
    Использует Context7 паттерн "Memory Management" для
    scalable storage и retrieval crypto trading experiences.
    """
    
    def __init__(self, config: NGUConfig):
        self.config = config
        self.capacity = config.episodic_memory_capacity
        self.embedding_dim = config.embedding_dim
        
        # FAISS index для быстрого поиска neighbors
        if config.faiss_gpu and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatL2(res, config.embedding_dim)
                logger.info("Using GPU FAISS index for episodic memory")
            except:
                self.index = faiss.IndexFlatL2(config.embedding_dim)
                logger.info("Using CPU FAISS index for episodic memory")
        else:
            self.index = faiss.IndexFlatL2(config.embedding_dim)
            logger.info("Using CPU FAISS index for episodic memory")
        
        # Memory storage
        self.embeddings = np.zeros((config.episodic_memory_capacity, config.embedding_dim), dtype=np.float32)
        self.rewards = np.zeros(config.episodic_memory_capacity, dtype=np.float32)
        self.episode_ids = np.zeros(config.episodic_memory_capacity, dtype=np.int32)
        self.timestamps = np.zeros(config.episodic_memory_capacity, dtype=np.float64)
        
        # Crypto-specific metadata
        self.market_regimes = np.zeros(config.episodic_memory_capacity, dtype=np.int32)
        self.portfolio_states = np.zeros((config.episodic_memory_capacity, 20), dtype=np.float32)
        self.risk_levels = np.zeros(config.episodic_memory_capacity, dtype=np.float32)
        
        # Memory management
        self.size = 0
        self.current_index = 0
        self.access_counts = np.zeros(config.episodic_memory_capacity, dtype=np.int32)
        self.last_access = np.zeros(config.episodic_memory_capacity, dtype=np.float64)
        
        logger.info(f"Episodic memory initialized with capacity {self.capacity}")
    
    def add(
        self,
        embedding: np.ndarray,
        reward: float,
        episode_id: int,
        market_regime: int = 0,
        portfolio_state: np.ndarray = None,
        risk_level: float = 0.0
    ) -> None:
        """
        Добавление нового experience в episodic memory.
        
        Args:
            embedding: State embedding для хранения
            reward: Intrinsic reward для данного состояния
            episode_id: ID эпизода
            market_regime: Тип рыночного режима
            portfolio_state: Состояние портфеля
            risk_level: Уровень риска
        """
        if self.size < self.capacity:
            # Простое добавление если есть место
            index = self.size
            self.size += 1
        else:
            # Replacement strategy
            if self.config.memory_replacement_strategy == "fifo":
                index = self.current_index
                self.current_index = (self.current_index + 1) % self.capacity
            elif self.config.memory_replacement_strategy == "lru":
                index = np.argmin(self.last_access)
            else:  # random
                index = np.random.randint(0, self.capacity)
        
        # Удаление старого embedding из index если он существует
        if index < self.index.ntotal:
            # FAISS не поддерживает удаление, поэтому пересоздаем index
            if self.size >= self.capacity:
                self._rebuild_index()
        
        # Сохранение нового experience
        self.embeddings[index] = embedding.astype(np.float32)
        self.rewards[index] = reward
        self.episode_ids[index] = episode_id
        self.timestamps[index] = time.time()
        
        if portfolio_state is not None:
            portfolio_state = portfolio_state[:20]  # Ограничение размера
            self.portfolio_states[index, :len(portfolio_state)] = portfolio_state
        
        self.market_regimes[index] = market_regime
        self.risk_levels[index] = risk_level
        self.access_counts[index] = 0
        self.last_access[index] = time.time()
        
        # Добавление в FAISS index
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
    
    def query_neighbors(
        self,
        embedding: np.ndarray,
        k: Optional[int] = None,
        exclude_episode: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск k ближайших neighbors для данного embedding.
        
        Args:
            embedding: Query embedding
            k: Количество neighbors (default: config.num_neighbors)
            exclude_episode: Исключить experiences из данного эпизода
            
        Returns:
            Tuple (distances, indices) ближайших neighbors
        """
        if k is None:
            k = min(self.config.num_neighbors, self.size)
        
        if self.size == 0:
            return np.array([]), np.array([])
        
        # Поиск neighbors
        distances, indices = self.index.search(
            embedding.reshape(1, -1).astype(np.float32), 
            min(k * 2, self.size)  # Больше кандидатов для фильтрации
        )
        
        distances = distances[0]
        indices = indices[0]
        
        # Фильтрация по episode_id если нужно
        if exclude_episode is not None:
            valid_mask = self.episode_ids[indices] != exclude_episode
            distances = distances[valid_mask]
            indices = indices[valid_mask]
        
        # Ограничение до k neighbors
        distances = distances[:k]
        indices = indices[:k]
        
        # Обновление access statistics
        current_time = time.time()
        for idx in indices:
            if idx < self.size:
                self.access_counts[idx] += 1
                self.last_access[idx] = current_time
        
        return distances, indices
    
    def compute_episodic_bonus(
        self,
        embedding: np.ndarray,
        episode_id: int
    ) -> float:
        """
        Вычисление episodic exploration bonus.
        
        Args:
            embedding: State embedding
            episode_id: Current episode ID
            
        Returns:
            Episodic bonus value
        """
        if self.size == 0:
            return self.config.episodic_bonus_maximum
        
        # Поиск ближайших neighbors (исключая текущий эпизод)
        distances, indices = self.query_neighbors(
            embedding,
            exclude_episode=episode_id
        )
        
        if len(distances) == 0:
            return self.config.episodic_bonus_maximum
        
        # Вычисление similarity-based bonus
        # Используем kernel density estimation
        similarities = np.exp(-distances / self.config.similarity_threshold)
        density_estimate = np.sum(similarities)
        
        # Episodic bonus обратно пропорционален density
        bonus = self.config.episodic_bonus_constant / np.sqrt(density_estimate + 1e-8)
        bonus = min(bonus, self.config.episodic_bonus_maximum)
        
        return bonus
    
    def get_market_regime_statistics(self, regime: int) -> Dict[str, float]:
        """Получение статистики для конкретного market regime."""
        if self.size == 0:
            return {}
        
        regime_mask = self.market_regimes[:self.size] == regime
        if not np.any(regime_mask):
            return {}
        
        regime_rewards = self.rewards[:self.size][regime_mask]
        regime_access = self.access_counts[:self.size][regime_mask]
        
        return {
            'count': np.sum(regime_mask),
            'avg_reward': np.mean(regime_rewards),
            'avg_access': np.mean(regime_access),
            'exploration_coverage': np.sum(regime_access > 0) / len(regime_access) if len(regime_access) > 0 else 0.0
        }
    
    def _rebuild_index(self) -> None:
        """Пересоздание FAISS index."""
        # Сохранение текущих embeddings
        current_embeddings = self.embeddings[:self.size].copy()
        
        # Пересоздание index
        if self.config.faiss_gpu and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatL2(res, self.config.embedding_dim)
            except:
                self.index = faiss.IndexFlatL2(self.config.embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(self.config.embedding_dim)
        
        # Добавление всех embeddings обратно
        if self.size > 0:
            self.index.add(current_embeddings.astype(np.float32))
    
    def save(self, filepath: str) -> None:
        """Сохранение episodic memory."""
        np.savez_compressed(
            filepath,
            embeddings=self.embeddings[:self.size],
            rewards=self.rewards[:self.size],
            episode_ids=self.episode_ids[:self.size],
            timestamps=self.timestamps[:self.size],
            market_regimes=self.market_regimes[:self.size],
            portfolio_states=self.portfolio_states[:self.size],
            risk_levels=self.risk_levels[:self.size],
            size=self.size,
            current_index=self.current_index
        )
        logger.info(f"Episodic memory saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Загрузка episodic memory."""
        data = np.load(filepath)
        
        self.size = int(data['size'])
        self.current_index = int(data['current_index'])
        
        self.embeddings[:self.size] = data['embeddings']
        self.rewards[:self.size] = data['rewards']
        self.episode_ids[:self.size] = data['episode_ids']
        self.timestamps[:self.size] = data['timestamps']
        self.market_regimes[:self.size] = data['market_regimes']
        self.portfolio_states[:self.size] = data['portfolio_states']
        self.risk_levels[:self.size] = data['risk_levels']
        
        # Пересоздание FAISS index
        self._rebuild_index()
        
        logger.info(f"Episodic memory loaded from {filepath}, size: {self.size}")


class StateEmbedder(nn.Module):
    """
    State embedder для создания compact representations.
    
    Применяет Context7 паттерн "Representation Learning" для
    efficient state encoding в crypto trading environments.
    """
    
    def __init__(self, config: NGUConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale feature extraction для crypto data
        self.feature_extractors = nn.ModuleDict({
            'market': nn.Sequential(
                nn.Linear(config.state_dim // 2, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.1),
                nn.Linear(128, 64)
            ),
            'portfolio': nn.Sequential(
                nn.Linear(config.state_dim // 4, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.1),
                nn.Linear(64, 32)
            ),
            'risk': nn.Sequential(
                nn.Linear(config.state_dim // 4, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 16)
            )
        })
        
        # Temporal context encoder
        self.temporal_encoder = nn.LSTM(
            input_size=112,  # 64 + 32 + 16
            hidden_size=config.embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Final embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        logger.info(f"State embedder initialized with output dim: {config.embedding_dim}")
    
    def forward(
        self,
        state: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Создание state embedding.
        
        Args:
            state: Current state [batch_size, state_dim]
            temporal_context: Historical states [batch_size, seq_len, state_dim]
            
        Returns:
            State embedding [batch_size, embedding_dim]
        """
        batch_size = state.size(0)
        
        # Разделение состояния на компоненты
        market_data = state[:, :self.config.state_dim // 2]
        portfolio_data = state[:, self.config.state_dim // 2:3 * self.config.state_dim // 4]
        risk_data = state[:, 3 * self.config.state_dim // 4:]
        
        # Feature extraction
        market_features = self.feature_extractors['market'](market_data)
        portfolio_features = self.feature_extractors['portfolio'](portfolio_data)
        risk_features = self.feature_extractors['risk'](risk_data)
        
        # Объединение features
        combined_features = torch.cat([market_features, portfolio_features, risk_features], dim=1)
        
        # Temporal encoding если есть context
        if temporal_context is not None:
            # Добавляем текущее состояние к контексту
            extended_context = torch.cat([temporal_context, combined_features.unsqueeze(1)], dim=1)
            lstm_out, _ = self.temporal_encoder(extended_context)
            # Используем последний выход
            temporal_embedding = lstm_out[:, -1]
        else:
            # Используем только текущее состояние
            temporal_embedding = combined_features
            # Пропускаем через LSTM для consistency
            lstm_out, _ = self.temporal_encoder(temporal_embedding.unsqueeze(1))
            temporal_embedding = lstm_out[:, -1]
        
        # Финальный embedding
        embedding = self.embedding_layer(temporal_embedding)
        
        # L2 normalization для stable similarity computation
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class NGUTrainer:
    """
    Trainer для Never Give Up agent с RND и episodic memory.
    
    Реализует Context7 паттерн "Multi-Component Learning" для
    coordinated training всех компонентов NGU.
    """
    
    def __init__(self, config: NGUConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Инициализация компонентов
        self.state_embedder = StateEmbedder(config).to(device)
        self.episodic_memory = EpisodicMemory(config)
        
        # RND components (импортируем из rnd.py)
        from .rnd import RNDConfig, RNDTrainer
        rnd_config = RNDConfig(
            state_dim=config.state_dim,
            target_network_dim=config.rnd_network_dim,
            predictor_network_dim=config.rnd_network_dim,
            learning_rate=config.rnd_learning_rate
        )
        self.rnd_trainer = RNDTrainer(rnd_config, device)
        
        # Optimizers
        self.embedder_optimizer = torch.optim.Adam(
            self.state_embedder.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # Exploration scheduling
        self.current_beta = config.exploration_beta_schedule[0]
        self.current_gamma = config.exploration_gamma_schedule[0]
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.temporal_contexts = defaultdict(lambda: deque(maxlen=config.temporal_context_length))
        
        # Metrics tracking
        self.episodic_bonuses = deque(maxlen=10000)
        self.rnd_bonuses = deque(maxlen=10000)
        self.total_intrinsic_rewards = deque(maxlen=10000)
        
        logger.info(f"NGU trainer initialized on device: {device}")
    
    def update_exploration_schedule(self, progress: float) -> None:
        """
        Обновление exploration hyperparameters по progress.
        
        Args:
            progress: Training progress [0, 1]
        """
        schedules = self.config.exploration_beta_schedule
        gammas = self.config.exploration_gamma_schedule
        
        if progress < 0.5:
            # Early exploration phase
            idx = 0
        elif progress < 0.8:
            # Mid exploration phase
            idx = 1
        else:
            # Late exploration phase
            idx = min(2, len(schedules) - 1)
        
        self.current_beta = schedules[idx]
        self.current_gamma = gammas[idx]
    
    def get_state_embedding(
        self,
        state: torch.Tensor,
        episode_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Получение state embedding с temporal context.
        
        Args:
            state: Current state
            episode_id: Episode ID для temporal context
            
        Returns:
            State embedding
        """
        with torch.no_grad():
            # Получение temporal context
            temporal_context = None
            if episode_id is not None and len(self.temporal_contexts[episode_id]) > 0:
                context_list = list(self.temporal_contexts[episode_id])
                temporal_context = torch.stack(context_list).unsqueeze(0).to(self.device)
            
            # Создание embedding
            embedding = self.state_embedder(state, temporal_context)
            
            # Обновление temporal context
            if episode_id is not None:
                # Сохраняем processed features для контекста
                market_data = state[:, :self.config.state_dim // 2]
                portfolio_data = state[:, self.config.state_dim // 2:3 * self.config.state_dim // 4]
                risk_data = state[:, 3 * self.config.state_dim // 4:]
                
                market_features = self.state_embedder.feature_extractors['market'](market_data)
                portfolio_features = self.state_embedder.feature_extractors['portfolio'](portfolio_data)
                risk_features = self.state_embedder.feature_extractors['risk'](risk_data)
                
                combined_features = torch.cat([market_features, portfolio_features, risk_features], dim=1)
                self.temporal_contexts[episode_id].append(combined_features.squeeze(0).cpu())
            
            return embedding
    
    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        episode_id: int,
        market_regime: int = 0,
        portfolio_state: Optional[np.ndarray] = None,
        risk_level: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Вычисление полного intrinsic reward через NGU.
        
        Args:
            state: Current state
            episode_id: Episode ID
            market_regime: Market regime type
            portfolio_state: Portfolio state
            risk_level: Risk level
            
        Returns:
            Tuple (total_intrinsic_reward, component_breakdown)
        """
        # RND intrinsic reward
        rnd_reward = self.rnd_trainer.compute_intrinsic_reward(state).item()
        
        # State embedding для episodic memory
        state_embedding = self.get_state_embedding(state, episode_id)
        embedding_np = state_embedding.cpu().numpy().squeeze()
        
        # Episodic exploration bonus
        episodic_bonus = self.episodic_memory.compute_episodic_bonus(
            embedding_np, episode_id
        )
        
        # Сохранение experience в episodic memory
        self.episodic_memory.add(
            embedding=embedding_np,
            reward=rnd_reward,
            episode_id=episode_id,
            market_regime=market_regime,
            portfolio_state=portfolio_state,
            risk_level=risk_level
        )
        
        # Комбинирование rewards
        total_intrinsic_reward = self.current_beta * rnd_reward + (1 - self.current_beta) * episodic_bonus
        
        # Сохранение для статистики
        self.episodic_bonuses.append(episodic_bonus)
        self.rnd_bonuses.append(rnd_reward)
        self.total_intrinsic_rewards.append(total_intrinsic_reward)
        
        component_breakdown = {
            'rnd_reward': rnd_reward,
            'episodic_bonus': episodic_bonus,
            'total_intrinsic': total_intrinsic_reward,
            'beta': self.current_beta,
            'gamma': self.current_gamma
        }
        
        return total_intrinsic_reward, component_breakdown
    
    def train_step(
        self,
        states: torch.Tensor,
        episode_ids: List[int]
    ) -> Dict[str, float]:
        """
        Выполнение training step для NGU компонентов.
        
        Args:
            states: Batch of states
            episode_ids: Episode IDs для каждого state
            
        Returns:
            Training metrics
        """
        batch_size = states.size(0)
        
        # Обновление RND
        rnd_metrics = self.rnd_trainer.train_step(states)
        
        # Обновление state embedder через contrastive learning
        self.embedder_optimizer.zero_grad()
        
        embeddings = []
        for i, episode_id in enumerate(episode_ids):
            embedding = self.get_state_embedding(states[i:i+1], episode_id)
            embeddings.append(embedding)
        
        embeddings = torch.cat(embeddings, dim=0)
        
        # Contrastive loss для улучшения embeddings
        # Positive pairs: states из одного эпизода
        # Negative pairs: states из разных эпизодов
        contrastive_loss = self._compute_contrastive_loss(embeddings, episode_ids)
        
        contrastive_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.state_embedder.parameters(), max_norm=1.0)
        self.embedder_optimizer.step()
        
        self.training_step += 1
        
        # Объединение метрик
        metrics = {
            **rnd_metrics,
            'contrastive_loss': contrastive_loss.item(),
            'episodic_bonus_mean': np.mean(list(self.episodic_bonuses)[-100:]) if self.episodic_bonuses else 0.0,
            'total_intrinsic_mean': np.mean(list(self.total_intrinsic_rewards)[-100:]) if self.total_intrinsic_rewards else 0.0,
            'memory_size': self.episodic_memory.size,
            'current_beta': self.current_beta,
            'current_gamma': self.current_gamma,
            'training_step': self.training_step
        }
        
        return metrics
    
    def _compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        episode_ids: List[int],
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Contrastive loss для embeddings."""
        batch_size = embeddings.size(0)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Mask для positive pairs (same episode)
        episode_tensor = torch.tensor(episode_ids, device=self.device)
        positive_mask = (episode_tensor.unsqueeze(0) == episode_tensor.unsqueeze(1)).float()
        
        # Убираем diagonal
        positive_mask = positive_mask - torch.eye(batch_size, device=self.device)
        
        # Contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean log-likelihood for positive pairs
        positive_log_prob = (positive_mask * log_prob).sum(dim=1)
        positive_counts = positive_mask.sum(dim=1)
        
        # Избегаем деления на ноль
        positive_counts = torch.clamp(positive_counts, min=1.0)
        loss = -(positive_log_prob / positive_counts).mean()
        
        return loss
    
    def reset_episode(self, episode_id: int) -> None:
        """Сброс данных эпизода."""
        if episode_id in self.temporal_contexts:
            self.temporal_contexts[episode_id].clear()
        self.episode_count += 1
    
    def save_checkpoint(self, filepath: str) -> None:
        """Сохранение checkpoint NGU."""
        checkpoint = {
            'state_embedder': self.state_embedder.state_dict(),
            'embedder_optimizer': self.embedder_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'current_beta': self.current_beta,
            'current_gamma': self.current_gamma
        }
        torch.save(checkpoint, filepath)
        
        # Сохранение episodic memory отдельно
        memory_filepath = filepath.replace('.pth', '_memory.npz')
        self.episodic_memory.save(memory_filepath)
        
        # Сохранение RND
        rnd_filepath = filepath.replace('.pth', '_rnd.pth')
        self.rnd_trainer.save_checkpoint(rnd_filepath)
        
        logger.info(f"NGU checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Загрузка checkpoint NGU."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.state_embedder.load_state_dict(checkpoint['state_embedder'])
        self.embedder_optimizer.load_state_dict(checkpoint['embedder_optimizer'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.current_beta = checkpoint['current_beta']
        self.current_gamma = checkpoint['current_gamma']
        
        # Загрузка episodic memory
        memory_filepath = filepath.replace('.pth', '_memory.npz')
        try:
            self.episodic_memory.load(memory_filepath)
        except FileNotFoundError:
            logger.warning(f"Episodic memory file not found: {memory_filepath}")
        
        # Загрузка RND
        rnd_filepath = filepath.replace('.pth', '_rnd.pth')
        try:
            self.rnd_trainer.load_checkpoint(rnd_filepath)
        except FileNotFoundError:
            logger.warning(f"RND checkpoint file not found: {rnd_filepath}")
        
        logger.info(f"NGU checkpoint loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение подробной статистики NGU."""
        stats = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'memory_size': self.episodic_memory.size,
            'current_beta': self.current_beta,
            'current_gamma': self.current_gamma,
            'episodic_bonuses': {
                'mean': np.mean(list(self.episodic_bonuses)) if self.episodic_bonuses else 0.0,
                'std': np.std(list(self.episodic_bonuses)) if self.episodic_bonuses else 0.0,
                'count': len(self.episodic_bonuses)
            },
            'rnd_bonuses': {
                'mean': np.mean(list(self.rnd_bonuses)) if self.rnd_bonuses else 0.0,
                'std': np.std(list(self.rnd_bonuses)) if self.rnd_bonuses else 0.0,
                'count': len(self.rnd_bonuses)
            },
            'total_intrinsic': {
                'mean': np.mean(list(self.total_intrinsic_rewards)) if self.total_intrinsic_rewards else 0.0,
                'std': np.std(list(self.total_intrinsic_rewards)) if self.total_intrinsic_rewards else 0.0,
                'count': len(self.total_intrinsic_rewards)
            }
        }
        
        # Добавление RND статистики
        rnd_stats = self.rnd_trainer.get_statistics()
        stats['rnd'] = rnd_stats
        
        return stats


def create_ngu_system(config: NGUConfig) -> NGUTrainer:
    """
    Factory function для создания NGU system.
    
    Args:
        config: NGU configuration
        
    Returns:
        Configured NGU trainer
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ngu_trainer = NGUTrainer(config, device)
    
    logger.info("NGU system created successfully")
    logger.info(f"State embedder parameters: {sum(p.numel() for p in ngu_trainer.state_embedder.parameters())}")
    logger.info(f"Episodic memory capacity: {config.episodic_memory_capacity}")
    
    return ngu_trainer


if __name__ == "__main__":
    # Пример использования NGU для crypto trading exploration
    config = NGUConfig(
        state_dim=256,
        action_dim=5,
        embedding_dim=64,
        episodic_memory_capacity=10000,
        num_neighbors=10
    )
    
    ngu_trainer = create_ngu_system(config)
    
    # Симуляция training
    batch_size = 32
    states = torch.randn(batch_size, config.state_dim)
    episode_ids = [i % 5 for i in range(batch_size)]  # 5 разных эпизодов
    
    # Training step
    metrics = ngu_trainer.train_step(states, episode_ids)
    print("Training metrics:", metrics)
    
    # Получение intrinsic reward
    single_state = torch.randn(1, config.state_dim)
    intrinsic_reward, breakdown = ngu_trainer.compute_intrinsic_reward(
        single_state, episode_id=0, market_regime=1
    )
    print(f"Intrinsic reward: {intrinsic_reward:.4f}")
    print("Breakdown:", breakdown)
    
    # Статистика
    stats = ngu_trainer.get_statistics()
    print("NGU Statistics:", stats)