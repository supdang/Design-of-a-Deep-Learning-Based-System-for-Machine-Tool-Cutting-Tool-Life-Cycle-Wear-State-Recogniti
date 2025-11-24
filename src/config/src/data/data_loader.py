#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载模块
负责加载和处理PHM2010数据集
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import scipy.io as sio
from ..utils.logger import get_module_logger
from ..config.config import DataConfig

logger = get_module_logger(__name__)

class PHM2010DataLoader:
    """PHM2010数据集加载器"""
    
    def __init__(self, data_dir: Path):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = DataConfig.SAMPLE_RATE
        self.sensor_channels = DataConfig.SENSOR_CHANNELS
        
    def load_mat_file(self, file_path: Path) -> Dict:
        """
        加载MAT格式文件
        
        Args:
            file_path: MAT文件路径
            
        Returns:
            加载的数据字典
        """
        try:
            data = sio.loadmat(file_path)
            logger.info(f"成功加载MAT文件: {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载MAT文件失败 {file_path}: {str(e)}")
            raise
    
    def load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """
        加载CSV格式文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            数据DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"成功加载CSV文件: {file_path}, 形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载CSV文件失败 {file_path}: {str(e)}")
            raise
    
    def extract_sensor_data(self, mat_data: Dict) -> np.ndarray:
        """
        从MAT数据中提取传感器数据
        
        Args:
            mat_data: MAT文件数据
            
        Returns:
            传感器数据数组 (channels, samples)
        """
        try:
            # PHM2010数据集中的传感器数据通常存储在'data'或'signal'字段中
            if 'data' in mat_data:
                raw_data = mat_data['data']
            elif 'signal' in mat_data:
                raw_data = mat_data['signal']
            elif 'X' in mat_data:
                raw_data = mat_data['X']
            else:
                # 如果没有找到标准字段，查找数值型数据
                numeric_keys = [k for k in mat_data.keys() 
                               if isinstance(mat_data[k], np.ndarray) 
                               and mat_data[k].dtype in [np.float64, np.float32]]
                if numeric_keys:
                    raw_data = mat_data[numeric_keys[0]]
                else:
                    raise ValueError("无法找到传感器数据")
            
            # 确保数据形状正确
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)
            elif raw_data.ndim > 2:
                raw_data = raw_data.reshape(raw_data.shape[0], -1)
            
            logger.info(f"提取传感器数据，形状: {raw_data.shape}")
            return raw_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"提取传感器数据失败: {str(e)}")
            raise
    
    def extract_wear_labels(self, mat_data: Dict) -> Optional[np.ndarray]:
        """
        从MAT数据中提取磨损标签
        
        Args:
            mat_data: MAT文件数据
            
        Returns:
            磨损标签数组，如果没有找到则返回None
        """
        try:
            # 查找可能的标签字段
            label_keys = ['wear', 'label', 'y', 'target', 'vbmax']
            
            for key in label_keys:
                if key in mat_data:
                    labels = mat_data[key].flatten()
                    logger.info(f"找到磨损标签，字段: {key}, 形状: {labels.shape}")
                    return labels.astype(np.float32)
            
            logger.warning("未找到磨损标签数据")
            return None
            
        except Exception as e:
            logger.error(f"提取磨损标签失败: {str(e)}")
            return None
    
    def create_wear_state_labels(self, wear_values: np.ndarray) -> np.ndarray:
        """
        根据磨损值创建磨损状态标签
        
        Args:
            wear_values: 磨损值数组
            
        Returns:
            磨损状态标签 (0:初期, 1:正常, 2:后期, 3:失效)
        """
        thresholds = DataConfig.WEAR_THRESHOLDS
        
        # 初始化标签数组
        state_labels = np.zeros(len(wear_values), dtype=np.int32)
        
        # 根据阈值分配状态
        state_labels[wear_values >= thresholds[2]] = 3  # 失效状态
        state_labels[(wear_values >= thresholds[1]) & (wear_values < thresholds[2])] = 2  # 后期磨损
        state_labels[(wear_values >= thresholds[0]) & (wear_values < thresholds[1])] = 1  # 正常磨损
        state_labels[wear_values < thresholds[0]] = 0  # 初期磨损
        
        logger.info(f"创建磨损状态标签，分布: {np.bincount(state_labels)}")
        return state_labels
    
    def load_dataset(self, dataset_name: str = "phm2010") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        加载完整数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            (sensor_data, labels, file_names) 元组
        """
        logger.info(f"开始加载数据集: {dataset_name}")
        
        all_sensor_data = []
        all_labels = []
        file_names = []
        
        # 查找数据文件
        data_files = list(self.data_dir.glob("**/*.mat")) + list(self.data_dir.glob("**/*.csv"))
        
        if not data_files:
            logger.warning(f"在 {self.data_dir} 中未找到数据文件")
            # 创建模拟数据用于演示
            return self._create_demo_data()
        
        for file_path in data_files:
            try:
                logger.info(f"处理文件: {file_path}")
                
                if file_path.suffix.lower() == '.mat':
                    mat_data = self.load_mat_file(file_path)
                    sensor_data = self.extract_sensor_data(mat_data)
                    wear_labels = self.extract_wear_labels(mat_data)
                    
                    # 如果没有磨损标签，创建模拟标签
                    if wear_labels is None:
                        wear_labels = np.random.uniform(0, 0.4, sensor_data.shape[1])
                    
                elif file_path.suffix.lower() == '.csv':
                    csv_data = self.load_csv_file(file_path)
                    # 假设CSV文件包含传感器数据列
                    sensor_cols = [col for col in csv_data.columns if any(sensor in col.lower() 
                                   for sensor in ['force', 'vibration', 'ae', 'acc'])]
                    
                    if sensor_cols:
                        sensor_data = csv_data[sensor_cols].values.T
                    else:
                        # 如果没有找到传感器列，使用所有数值列
                        numeric_cols = csv_data.select_dtypes(include=[np.number]).columns
                        sensor_data = csv_data[numeric_cols].values.T
                    
                    # 创建模拟磨损标签
                    wear_labels = np.random.uniform(0, 0.4, sensor_data.shape[1])
                
                # 创建状态标签
                state_labels = self.create_wear_state_labels(wear_labels)
                
                # 确保数据维度匹配
                if sensor_data.shape[1] == len(state_labels):
                    all_sensor_data.append(sensor_data)
                    all_labels.append(state_labels)
                    file_names.append(file_path.name)
                else:
                    logger.warning(f"数据维度不匹配: {file_path}")
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {str(e)}")
                continue
        
        if not all_sensor_data:
            logger.warning("没有成功加载任何数据文件，创建演示数据")
            return self._create_demo_data()
        
        # 合并所有数据
        sensor_data = np.concatenate(all_sensor_data, axis=1) if len(all_sensor_data) > 1 else all_sensor_data[0]
        labels = np.concatenate(all_labels) if len(all_labels) > 1 else all_labels[0]
        
        logger.info(f"数据集加载完成，传感器数据形状: {sensor_data.shape}, 标签形状: {labels.shape}")
        return sensor_data, labels, file_names
    
    def _create_demo_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建演示数据
        
        Returns:
            演示数据元组
        """
        logger.info("创建演示数据")
        
        # 创建模拟的传感器数据
        n_samples = 10000
        n_channels = len(self.sensor_channels)
        
        sensor_data = np.random.randn(n_channels, n_samples).astype(np.float32)
        
        # 添加一些趋势来模拟磨损过程
        trend = np.linspace(0, 1, n_samples)
        for i in range(n_channels):
            sensor_data[i] += trend * np.random.uniform(0.5, 2.0)
            sensor_data[i] += np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.5
        
        # 创建模拟的磨损标签
        wear_progression = np.linspace(0, 0.4, n_samples)
        labels = self.create_wear_state_labels(wear_progression)
        
        file_names = ["demo_data.mat"]
        
        logger.info(f"演示数据创建完成，传感器数据形状: {sensor_data.shape}, 标签形状: {labels.shape}")
        return sensor_data, labels, file_names
    
    def get_file_info(self, file_path: Path) -> Dict:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        try:
            file_stat = file_path.stat()
            
            if file_path.suffix.lower() == '.mat':
                mat_data = self.load_mat_file(file_path)
                sensor_data = self.extract_sensor_data(mat_data)
                wear_labels = self.extract_wear_labels(mat_data)
                
                info = {
                    'file_name': file_path.name,
                    'file_size': file_stat.st_size,
                    'modified_time': file_stat.st_mtime,
                    'data_shape': sensor_data.shape,
                    'has_labels': wear_labels is not None,
                    'data_type': 'MATLAB'
                }
                
                if wear_labels is not None:
                    info['label_range'] = [float(wear_labels.min()), float(wear_labels.max())]
                    info['label_distribution'] = np.bincount(self.create_wear_state_labels(wear_labels)).tolist()
            
            elif file_path.suffix.lower() == '.csv':
                df = self.load_csv_file(file_path)
                info = {
                    'file_name': file_path.name,
                    'file_size': file_stat.st_size,
                    'modified_time': file_stat.st_mtime,
                    'data_shape': df.shape,
                    'columns': df.columns.tolist(),
                    'data_type': 'CSV'
                }
            
            else:
                info = {
                    'file_name': file_path.name,
                    'file_size': file_stat.st_size,
                    'modified_time': file_stat.st_mtime,
                    'data_type': 'Unknown'
                }
            
            return info
            
        except Exception as e:
            logger.error(f"获取文件信息失败 {file_path}: {str(e)}")
            return {
                'file_name': file_path.name,
                'error': str(e)
            }