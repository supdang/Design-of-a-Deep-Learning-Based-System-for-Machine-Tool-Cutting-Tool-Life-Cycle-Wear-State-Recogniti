#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本
用于验证系统的基本功能
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.config import DataConfig, ModelConfig
from data.data_loader import PHM2010DataLoader
from data.preprocessor import SignalPreprocessor
from models.lstm_model import LSTMToolWearModel, ToolWearClassifier
from utils.logger import setup_logger

def test_imports():
    """测试模块导入"""
    print("✓ 所有模块导入成功")

def test_config():
    """测试配置系统"""
    print(f"✓ 数据配置正常: 采样频率={DataConfig.SAMPLE_RATE}Hz")
    print(f"✓ 模型配置正常: 隐藏层大小={ModelConfig.HIDDEN_SIZE}")

def test_data_loader():
    """测试数据加载器"""
    try:
        # 创建数据目录
        data_dir = Path(__file__).parent / 'data'
        data_dir.mkdir(exist_ok=True)
        
        # 创建演示数据
        demo_data = np.random.randn(7, 10000).astype(np.float32)
        demo_labels = np.random.randint(0, 4, 10000)
        
        # 保存为numpy文件
        np.savez(data_dir / 'demo_data.npz', 
                sensor_data=demo_data, 
                labels=demo_labels)
        
        # 测试数据加载
        data_loader = PHM2010DataLoader(data_dir)
        sensor_data, labels, file_names = data_loader.load_dataset()
        
        print(f"✓ 数据加载成功: 数据形状={sensor_data.shape}, 标签形状={labels.shape}")
        return sensor_data, labels
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None, None

def test_preprocessor(sensor_data, labels):
    """测试数据预处理器"""
    try:
        preprocessor = SignalPreprocessor()
        
        # 测试小波去噪
        denoised_data = preprocessor.wavelet_denoising(sensor_data[:, :1024])
        print(f"✓ 小波去噪成功: 输入形状={sensor_data[:, :1024].shape}, 输出形状={denoised_data.shape}")
        
        # 测试归一化
        normalized_data = preprocessor.normalize_signal(sensor_data[:, :1024])
        print(f"✓ 数据归一化成功: 范围=[{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
        
        # 测试滑动窗口
        windows, window_labels = preprocessor.create_sliding_windows(
            sensor_data[:, :2048], labels[:2048])
        print(f"✓ 滑动窗口创建成功: 窗口形状={windows.shape}, 标签形状={window_labels.shape}")
        
        return windows, window_labels
        
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return None, None

def test_lstm_model(windows, labels):
    """测试LSTM模型"""
    try:
        # 创建模型
        model = LSTMToolWearModel(
            input_size=windows.shape[1],
            hidden_size=64,  # 测试时使用较小的隐藏层
            num_layers=2,
            num_classes=4,
            dropout_rate=0.2
        )
        
        # 测试前向传播
        test_input = torch.randn(2, windows.shape[2], windows.shape[1])
        output = model(test_input)
        
        print(f"✓ LSTM模型创建成功: 输入形状={test_input.shape}, 输出形状={output.shape}")
        
        # 测试分类器
        classifier = ToolWearClassifier(model=model)
        
        # 分割数据
        train_size = int(0.8 * len(windows))
        train_data = (windows[:train_size], labels[:train_size])
        val_data = (windows[train_size:], labels[train_size:])
        
        # 快速训练测试
        history = classifier.train(
            train_data, val_data,
            batch_size=16,
            num_epochs=5,  # 测试时只训练5轮
            learning_rate=0.001
        )
        
        final_acc = history['val_acc'][-1]
        print(f"✓ 模型训练成功: 最终验证准确率={final_acc:.4f}")
        
        # 测试预测
        predictions, probabilities = classifier.predict(val_data[0][:10])
        print(f"✓ 模型预测成功: 预测结果形状={predictions.shape}, 概率形状={probabilities.shape}")
        
        return classifier
        
    except Exception as e:
        print(f"✗ LSTM模型测试失败: {e}")
        return None

def test_gui():
    """测试GUI界面"""
    try:
        app = QApplication(sys.argv)
        from gui.main_window import MainWindow
        
        # 创建主窗口但不显示
        window = MainWindow(debug_mode=True)
        
        print("✓ GUI界面创建成功")
        return True
        
    except Exception as e:
        print(f"✗ GUI界面测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("刀具磨损诊断系统测试")
    print("=" * 60)
    
    # 设置日志
    logger = setup_logger(log_level="INFO")
    
    # 测试计数
    total_tests = 0
    passed_tests = 0
    
    # 1. 测试模块导入
    total_tests += 1
    try:
        test_imports()
        passed_tests += 1
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
    
    # 2. 测试配置系统
    total_tests += 1
    try:
        test_config()
        passed_tests += 1
    except Exception as e:
        print(f"✗ 配置系统测试失败: {e}")
    
    # 3. 测试数据加载器
    total_tests += 1
    sensor_data, labels = test_data_loader()
    if sensor_data is not None:
        passed_tests += 1
    
    # 4. 测试数据预处理器
    if sensor_data is not None:
        total_tests += 1
        windows, window_labels = test_preprocessor(sensor_data, labels)
        if windows is not None:
            passed_tests += 1
    else:
        windows, window_labels = None, None
    
    # 5. 测试LSTM模型
    if windows is not None:
        total_tests += 1
        classifier = test_lstm_model(windows, window_labels)
        if classifier is not None:
            passed_tests += 1
    else:
        classifier = None
    
    # 6. 测试GUI界面
    total_tests += 1
    if test_gui():
        passed_tests += 1
    
    # 测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"测试通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！系统运行正常。")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests}个测试失败，请检查相关问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
