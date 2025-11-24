#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM模型模块
实现基于LSTM的刀具磨损状态识别模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import get_module_logger
from ..config.config import ModelConfig

logger = get_module_logger(__name__)


class ToolWearDataset(Dataset):
    """刀具磨损数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        初始化数据集

        Args:
            features: 特征数据 (samples, timesteps, features)
            labels: 标签数据 (samples,)
        """
        # 确保数据形状正确
        if features.ndim == 2:
            # 假设形状为 (samples, features*timesteps)
            features = features.reshape(features.shape[0], ModelConfig.INPUT_SIZE, -1)
            features = features.transpose(0, 2, 1)  # 转换为 (samples, timesteps, features)

        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class LSTMToolWearModel(nn.Module):
    """LSTM刀具磨损识别模型"""

    def __init__(self,
                 input_size: int = ModelConfig.INPUT_SIZE,
                 hidden_size: int = ModelConfig.HIDDEN_SIZE,
                 num_layers: int = ModelConfig.NUM_LAYERS,
                 num_classes: int = ModelConfig.NUM_CLASSES,
                 dropout_rate: float = ModelConfig.DROPOUT_RATE):
        """
        初始化LSTM模型

        Args:
            input_size: 输入特征维度 (应为7，对应7个传感器通道)
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            num_classes: 分类数
            dropout_rate: Dropout比率
        """
        super(LSTMToolWearModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 遗忘门偏置初始化为1
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1)

        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, seq_len, input_size)

        Returns:
            输出张量 (batch_size, num_classes)
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 使用最后一个时间步的输出
        output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # 全连接层
        output = self.fc(output)  # (batch_size, num_classes)

        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取LSTM特征

        Args:
            x: 输入张量

        Returns:
            LSTM特征张量
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        return lstm_out[:, -1, :]


class ToolWearClassifier:
    """刀具磨损分类器"""

    def __init__(self,
                 model: Optional[LSTMToolWearModel] = None,
                 device: Optional[str] = None):
        """
        初始化分类器

        Args:
            model: LSTM模型实例
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"使用设备: {self.device}")

        # 初始化模型
        if model is None:
            self.model = LSTMToolWearModel()
        else:
            self.model = model

        self.model.to(self.device)

        # 初始化优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # 最佳模型
        self.best_val_acc = 0.0
        self.best_model_state = None

    def setup_optimizer(self,
                        learning_rate: float = ModelConfig.LEARNING_RATE,
                        weight_decay: float = ModelConfig.WEIGHT_DECAY):
        """
        设置优化器

        Args:
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        logger.info(f"优化器设置完成，学习率: {learning_rate}, 权重衰减: {weight_decay}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            (avg_loss, avg_acc) 元组
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            (avg_loss, avg_acc) 元组
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def train(self,
              train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray],
              batch_size: int = ModelConfig.BATCH_SIZE,
              num_epochs: int = ModelConfig.NUM_EPOCHS,
              learning_rate: float = ModelConfig.LEARNING_RATE,
              patience: int = ModelConfig.PATIENCE) -> Dict:
        """
        训练模型

        Args:
            train_data: 训练数据 (features, labels)
            val_data: 验证数据 (features, labels)
            batch_size: 批量大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值

        Returns:
            训练历史字典
        """
        # 设置优化器
        self.setup_optimizer(learning_rate)

        # 创建数据加载器
        train_dataset = ToolWearDataset(train_data[0], train_data[1])
        val_dataset = ToolWearDataset(val_data[0], val_data[1])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"开始训练，训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

        # 训练循环
        best_epoch = 0
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 更新学习率
            if self.scheduler:
                self.scheduler.step(val_loss)

            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)

            # 保存最佳模型
            if val_acc > self.best_val_acc + ModelConfig.MIN_DELTA:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - "
                            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 早停检查
            if no_improve_epochs >= patience:
                logger.info(f"早停触发，最佳验证准确率: {self.best_val_acc:.4f} at epoch {best_epoch}")
                break

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        logger.info(f"训练完成，最佳验证准确率: {self.best_val_acc:.4f}")
        return self.train_history

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测

        Args:
            features: 输入特征 (samples, timesteps, features)

        Returns:
            (predictions, probabilities) 元组
        """
        # 确保数据形状正确
        if features.ndim == 2:
            features = features.reshape(features.shape[0], DataConfig.WINDOW_SIZE, 7)

        self.model.eval()

        with torch.no_grad():
            # 转换为张量
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)

            features = features.to(self.device)

            # 预测
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """
        评估模型

        Args:
            test_data: 测试数据 (features, labels)

        Returns:
            评估结果字典
        """
        features, labels = test_data

        # 确保数据形状正确
        if features.ndim == 2:
            features = features.reshape(features.shape[0], DataConfig.WINDOW_SIZE, 7)

        predictions, probabilities = self.predict(features)

        # 计算评估指标
        accuracy = accuracy_score(labels, predictions)

        # 分类报告
        class_report = classification_report(
            labels, predictions,
            target_names=['初期磨损', '正常磨损', '后期磨损', '失效状态'],
            output_dict=True
        )

        # 混淆矩阵
        conf_matrix = confusion_matrix(labels, predictions)

        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': labels
        }

        logger.info(f"模型评估完成，准确率: {accuracy:.4f}")
        return results

    def save_model(self, file_path: str):
        """
        保存模型

        Args:
            file_path: 保存路径
        """
        try:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'num_classes': self.model.num_classes,
                    'dropout_rate': self.model.dropout_rate
                },
                'train_history': self.train_history,
                'best_val_acc': self.best_val_acc
            }

            torch.save(save_dict, file_path)
            logger.info(f"模型已保存到: {file_path}")

        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            raise

    def load_model(self, file_path: str):
        """
        加载模型

        Args:
            file_path: 模型文件路径
        """
        try:
            checkpoint = torch.load(file_path, map_location=self.device)

            # 重建模型
            config = checkpoint['model_config']
            self.model = LSTMToolWearModel(**config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)

            # 加载训练历史
            if 'train_history' in checkpoint:
                self.train_history = checkpoint['train_history']

            if 'best_val_acc' in checkpoint:
                self.best_val_acc = checkpoint['best_val_acc']

            logger.info(f"模型已从 {file_path} 加载")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        绘制训练历史

        Args:
            save_path: 保存路径
        """
        if not self.train_history['train_loss']:
            logger.warning("没有训练历史可以绘制")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='训练损失')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='训练准确率')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('训练准确率曲线')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图已保存到: {save_path}")

        plt.show()

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: Optional[str] = None):
        """
        绘制混淆矩阵

        Args:
            conf_matrix: 混淆矩阵
            save_path: 保存路径
        """
        plt.figure(figsize=(8, 6))

        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['初期磨损', '正常磨损', '后期磨损', '失效状态'],
                    yticklabels=['初期磨损', '正常磨损', '后期磨损', '失效状态'])

        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵图已保存到: {save_path}")

        plt.show()