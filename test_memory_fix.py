#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试内存管理和断点恢复修复
"""

import torch
import gc
import numpy as np
from pathlib import Path

# 测试CPU内存管理
def test_cpu_memory_management():
    """测试CPU内存管理改进"""
    print("测试CPU内存管理...")
    
    # 模拟大张量创建和清理
    large_tensor = torch.randn(1000, 1000, 100)  # 约3.8GB张量
    print(f"创建大张量后内存占用: {large_tensor.element_size() * large_tensor.nelement() / 1024**3:.2f} GB")
    
    # 删除张量并清理内存
    del large_tensor
    gc.collect()
    print("删除张量并执行垃圾回收后")
    
    # 创建较小的测试数据
    small_tensor = torch.randn(100, 100, 100)  # 约38MB张量
    print(f"创建小张量后内存占用: {small_tensor.element_size() * small_tensor.nelement() / 1024**3:.2f} GB")
    
    del small_tensor
    gc.collect()
    print("CPU内存管理测试完成")

# 测试断点恢复逻辑
def test_checkpoint_logic():
    """测试断点恢复逻辑"""
    print("\n测试断点恢复逻辑...")
    
    # 模拟从断点恢复的场景
    start_epoch = 2  # 假设已经训练了2轮
    best_epoch = 1   # 最佳模型在第1轮
    no_improve_epochs = 1  # 从上次最佳后已经1轮没有改善
    
    print(f"从断点恢复 - 当前轮数: {start_epoch}, 最佳轮数: {best_epoch}, 无改善轮数: {no_improve_epochs}")
    print(f"下一轮将从第 {start_epoch + 1} 轮开始训练")
    
    # 模拟训练循环
    total_epochs = 10
    for epoch in range(start_epoch, total_epochs):
        print(f"正在训练第 {epoch + 1} 轮...")
        # 在每轮结束时保存断点
        print(f"  -> 保存断点，当前轮数: {epoch}, 最佳轮数: {best_epoch}")
        
        if epoch >= start_epoch + 2:  # 模拟找到新的最佳模型
            best_epoch = epoch
            no_improve_epochs = 0
            print(f"  -> 发现新最佳模型，最佳轮数更新为: {best_epoch}")
        else:
            no_improve_epochs += 1
            print(f"  -> 无改善轮数: {no_improve_epochs}")
    
    print("断点恢复逻辑测试完成")

if __name__ == "__main__":
    test_cpu_memory_management()
    test_checkpoint_logic()
    print("\n所有测试完成！")