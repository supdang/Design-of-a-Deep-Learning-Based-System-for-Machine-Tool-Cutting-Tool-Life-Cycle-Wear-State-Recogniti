#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用训练脚本
支持CPU和GPU训练，以及断点保存和恢复功能
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import argparse

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config.config import DataConfig, ModelConfig
from src.data.data_loader import PHM2010DataLoader
from src.data.preprocessor import SignalPreprocessor
from src.models.lstm_model import ToolWearClassifier
from src.utils.logger import setup_logger


def main():
    """主训练函数"""
    print("=" * 60)
    print("刀具磨损状态识别模型 - 通用训练（支持断点）")
    print("=" * 60)
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='通用训练脚本（支持断点）')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                       help='训练设备 (cpu, cuda, auto)')
    parser.add_argument('--checkpoint-path', type=str, 
                       default='checkpoints/training_checkpoint.pth',
                       help='断点文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='从断点恢复训练')
    parser.add_argument('--epochs', type=int, default=ModelConfig.NUM_EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=ModelConfig.BATCH_SIZE,
                       help='批量大小')
    parser.add_argument('--lr', type=float, default=ModelConfig.LEARNING_RATE,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(log_level="INFO")
    logger.info("开始通用训练流程（支持断点）")
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    # 检查CUDA可用性
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，回退到CPU")
        device = 'cpu'
    
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 加载数据
        print("1. 加载数据...")
        data_dir = Path(__file__).parent / 'data'
        data_loader = PHM2010DataLoader(data_dir)
        sensor_data, labels, file_names = data_loader.load_dataset()
        
        print(f"   数据形状: {sensor_data.shape}")
        print(f"   标签形状: {labels.shape}")
        print(f"   文件名: {file_names}")
        
        # 2. 数据预处理
        print("\n2. 数据预处理...")
        preprocessor = SignalPreprocessor()
        
        # 应用预处理流水线
        processed_data, processed_labels = preprocessor.preprocess_pipeline(
            signal_data=sensor_data,
            labels=labels,
            apply_denoising=True,
            apply_filtering=True,
            apply_normalization=True,
            apply_outlier_removal=True,
            create_windows=True  # 创建滑动窗口
        )
        
        print(f"   预处理后数据形状: {processed_data.shape}")
        print(f"   预处理后标签形状: {processed_labels.shape}")
        
        # 3. 数据分割
        print("\n3. 数据分割...")
        # 如果是滑动窗口数据，直接使用
        if len(processed_data.shape) == 3:  # (n_windows, window_size, n_channels)
            X = processed_data
            y = processed_labels
        else:
            # 如果不是窗口格式，需要手动创建
            X = sensor_data.T.reshape(-1, DataConfig.WINDOW_SIZE, 7)
            y = labels[:X.shape[0]]
        
        # 确保数据维度正确
        print(f"   最终特征形状: {X.shape}")
        print(f"   最终标签形状: {y.shape}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 进一步分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"   训练集形状: {X_train.shape}")
        print(f"   验证集形状: {X_val.shape}")
        print(f"   测试集形状: {X_test.shape}")
        
        # 4. 创建模型
        print("\n4. 创建LSTM模型...")
        classifier = ToolWearClassifier(device=device)
        print(f"   模型创建成功，使用设备: {classifier.device}")
        
        # 5. 确保断点目录存在
        checkpoint_dir = Path(args.checkpoint_path).parent
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 6. 训练模型
        print("\n5. 开始训练...")
        print(f"   使用设备: {classifier.device}")
        print(f"   训练轮数: {args.epochs}")
        print(f"   批量大小: {args.batch_size}")
        print(f"   学习率: {args.lr}")
        print(f"   断点路径: {args.checkpoint_path}")
        print(f"   从断点恢复: {args.resume}")
        
        # 训练模型
        history = classifier.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=ModelConfig.PATIENCE,
            checkpoint_path=args.checkpoint_path,
            resume_from_checkpoint=args.resume
        )
        
        print(f"\n   训练完成！最佳验证准确率: {classifier.best_val_acc:.4f}")
        
        # 6. 评估模型
        print("\n6. 评估模型...")
        test_results = classifier.evaluate((X_test, y_test))
        print(f"   测试准确率: {test_results['accuracy']:.4f}")
        
        # 7. 保存最终模型
        print("\n7. 保存模型...")
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / ModelConfig.MODEL_NAME
        classifier.save_model(str(model_path))
        print(f"   模型已保存到: {model_path}")
        
        # 8. 保存最佳模型
        best_model_path = model_dir / ModelConfig.BEST_MODEL_NAME
        classifier.save_model(str(best_model_path))
        print(f"   最佳模型已保存到: {best_model_path}")
        
        # 9. 绘制训练历史
        print("\n8. 绘制训练历史...")
        plot_dir = Path(__file__).parent / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        classifier.plot_training_history(save_path=plot_dir / 'training_history.png')
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"模型已保存到: {model_dir}")
        print(f"图表已保存到: {plot_dir}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")
        return 1


def quick_train():
    """快速训练模式（用于测试）"""
    print("=" * 60)
    print("快速训练模式（支持断点，测试用）")
    print("=" * 60)
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='快速训练模式（支持断点）')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                       help='训练设备 (cpu, cuda, auto)')
    parser.add_argument('--checkpoint-path', type=str, 
                       default='checkpoints/quick_training_checkpoint.pth',
                       help='断点文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='从断点恢复训练')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(log_level="INFO")
    logger.info("开始快速训练流程（支持断点）")
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    # 检查CUDA可用性
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，回退到CPU")
        device = 'cpu'
    
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建演示数据
        print("1. 创建演示数据...")
        n_samples = 2000
        n_channels = 7  # 7个传感器通道
        window_size = 128  # 使用较小的窗口以加快训练
        
        # 创建模拟传感器数据
        X = np.random.randn(n_samples, window_size, n_channels).astype(np.float32)
        
        # 添加一些模式到数据中以模拟磨损状态
        for i in range(n_channels):
            trend = np.linspace(0, 1, window_size)
            X[:, :, i] += trend * np.random.uniform(-0.5, 0.5)
            X[:, :, i] += np.sin(np.linspace(0, 4*np.pi, window_size)) * 0.2
        
        # 创建标签 (0:初期磨损, 1:正常磨损, 2:后期磨损, 3:失效状态)
        y = np.random.randint(0, 4, n_samples).astype(np.int64)
        
        print(f"   特征形状: {X.shape}")
        print(f"   标签形状: {y.shape}")
        
        # 分割数据
        split_idx1 = int(0.6 * n_samples)
        split_idx2 = int(0.8 * n_samples)
        
        X_train, y_train = X[:split_idx1], y[:split_idx1]
        X_val, y_val = X[split_idx1:split_idx2], y[split_idx1:split_idx2]
        X_test, y_test = X[split_idx2:], y[split_idx2:]
        
        print(f"   训练集: {X_train.shape}")
        print(f"   验证集: {X_val.shape}")
        print(f"   测试集: {X_test.shape}")
        
        # 创建模型（简化版以加快训练）
        print("\n2. 创建简化LSTM模型...")
        from src.models.lstm_model import LSTMToolWearModel
        model = LSTMToolWearModel(
            input_size=n_channels,
            hidden_size=64,  # 使用较小的隐藏层
            num_layers=1,    # 使用较少层数
            num_classes=4,
            dropout_rate=0.1
        )
        
        classifier = ToolWearClassifier(model=model, device=device)
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 确保断点目录存在
        checkpoint_dir = Path(args.checkpoint_path).parent
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 快速训练
        print("\n3. 开始快速训练...")
        print(f"   设备: {classifier.device}")
        print(f"   训练轮数: {args.epochs}")
        print(f"   批量大小: {args.batch_size}")
        print(f"   断点路径: {args.checkpoint_path}")
        print(f"   从断点恢复: {args.resume}")
        
        history = classifier.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            patience=5,
            checkpoint_path=args.checkpoint_path,
            resume_from_checkpoint=args.resume
        )
        
        print(f"\n   快速训练完成！最佳验证准确率: {classifier.best_val_acc:.4f}")
        
        # 评估
        print("\n4. 评估模型...")
        test_results = classifier.evaluate((X_test, y_test))
        print(f"   测试准确率: {test_results['accuracy']:.4f}")
        
        # 保存模型
        print("\n5. 保存模型...")
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "quick_train_model.pth"
        classifier.save_model(str(model_path))
        print(f"   模型已保存到: {model_path}")
        
        print("\n" + "=" * 60)
        print("快速训练完成！")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"快速训练过程中发生错误: {str(e)}", exc_info=True)
        print(f"\n错误: {str(e)}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='通用训练脚本（支持断点）')
    parser.add_argument('--quick', action='store_true', help='使用快速训练模式（测试用）')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                       help='训练设备 (cpu, cuda, auto)')
    parser.add_argument('--checkpoint-path', type=str, 
                       default='checkpoints/training_checkpoint.pth',
                       help='断点文件路径')
    parser.add_argument('--resume', action='store_true',
                       help='从断点恢复训练')
    
    args = parser.parse_args()
    
    if args.quick:
        exit_code = quick_train()
    else:
        exit_code = main()
    
    sys.exit(exit_code)