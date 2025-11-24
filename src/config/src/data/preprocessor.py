#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
实现信号去噪、归一化、时间序列切分等预处理功能
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, List
import pywt
from ..utils.logger import get_module_logger
from ..config.config import PreprocessConfig, DataConfig

logger = get_module_logger(__name__)


class SignalPreprocessor:
    """信号预处理器"""

    def __init__(self):
        """初始化预处理器"""
        self.window_size = DataConfig.WINDOW_SIZE
        self.step_size = DataConfig.STEP_SIZE
        self.wavelet_name = PreprocessConfig.WAVELET_NAME
        self.decomposition_level = PreprocessConfig.DECOMPOSITION_LEVEL

        # 初始化scaler
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.standard_scaler = StandardScaler()

    def wavelet_denoising(self, signal_data: np.ndarray) -> np.ndarray:
        """
        使用小波变换进行信号去噪

        Args:
            signal_data: 输入信号数组 (channels, samples)

        Returns:
            去噪后的信号数组
        """
        try:
            denoised_data = np.zeros_like(signal_data)

            for channel in range(signal_data.shape[0]):
                # 小波分解
                coeffs = pywt.wavedec(
                    signal_data[channel],
                    self.wavelet_name,
                    level=self.decomposition_level
                )

                # 计算阈值
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(signal_data[channel])))

                # 软阈值处理
                coeffs_thresh = list(coeffs)
                for i in range(1, len(coeffs)):
                    coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

                # 小波重构
                denoised_data[channel] = pywt.waverec(coeffs_thresh, self.wavelet_name)

                # 确保长度一致
                if len(denoised_data[channel]) != len(signal_data[channel]):
                    denoised_data[channel] = denoised_data[channel][:len(signal_data[channel])]

            logger.info(f"小波去噪完成，原始形状: {signal_data.shape}, 处理后形状: {denoised_data.shape}")
            return denoised_data.astype(np.float32)

        except Exception as e:
            logger.error(f"小波去噪失败: {str(e)}")
            return signal_data

    def butterworth_filter(self, signal_data: np.ndarray,
                           lowcut: Optional[float] = None,
                           highcut: Optional[float] = None,
                           fs: int = DataConfig.SAMPLE_RATE) -> np.ndarray:
        """
        使用巴特沃斯滤波器进行信号滤波

        Args:
            signal_data: 输入信号数组
            lowcut: 低频截止频率 (Hz)
            highcut: 高频截止频率 (Hz)
            fs: 采样频率 (Hz)

        Returns:
            滤波后的信号数组
        """
        try:
            nyquist = 0.5 * fs

            if lowcut is not None and highcut is not None:
                # 带通滤波
                low = lowcut / nyquist
                high = highcut / nyquist
                b, a = signal.butter(4, [low, high], btype='band')
            elif lowcut is not None:
                # 高通滤波
                low = lowcut / nyquist
                b, a = signal.butter(4, low, btype='high')
            elif highcut is not None:
                # 低通滤波
                high = highcut / nyquist
                b, a = signal.butter(4, high, btype='low')
            else:
                return signal_data

            filtered_data = np.zeros_like(signal_data)
            for channel in range(signal_data.shape[0]):
                filtered_data[channel] = signal.filtfilt(b, a, signal_data[channel])

            logger.info(f"巴特沃斯滤波完成，滤波器类型: {self._get_filter_type(lowcut, highcut)}")
            return filtered_data.astype(np.float32)

        except Exception as e:
            logger.error(f"巴特沃斯滤波失败: {str(e)}")
            return signal_data

    def _get_filter_type(self, lowcut: Optional[float], highcut: Optional[float]) -> str:
        """获取滤波器类型"""
        if lowcut is not None and highcut is not None:
            return f"带通滤波 ({lowcut}-{highcut} Hz)"
        elif lowcut is not None:
            return f"高通滤波 ({lowcut} Hz)"
        elif highcut is not None:
            return f"低通滤波 ({highcut} Hz)"
        else:
            return "无滤波"

    def normalize_signal(self, signal_data: np.ndarray,
                         method: str = 'minmax') -> np.ndarray:
        """
        信号归一化处理

        Args:
            signal_data: 输入信号数组
            method: 归一化方法 ('minmax', 'standard', 'zscore')

        Returns:
            归一化后的信号数组
        """
        try:
            normalized_data = np.zeros_like(signal_data)

            for channel in range(signal_data.shape[0]):
                signal_channel = signal_data[channel].reshape(-1, 1)

                if method == 'minmax':
                    # Min-Max归一化到[0,1]
                    normalized_data[channel] = self.minmax_scaler.fit_transform(signal_channel).flatten()
                elif method == 'standard':
                    # 标准化到均值为0，方差为1
                    normalized_data[channel] = self.standard_scaler.fit_transform(signal_channel).flatten()
                elif method == 'zscore':
                    # Z-score标准化
                    normalized_data[channel] = zscore(signal_data[channel])
                else:
                    raise ValueError(f"不支持的归一化方法: {method}")

            logger.info(f"信号归一化完成，方法: {method}")
            return normalized_data.astype(np.float32)

        except Exception as e:
            logger.error(f"信号归一化失败: {str(e)}")
            return signal_data

    def create_sliding_windows(self, signal_data: np.ndarray,
                               labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用滑动窗口创建时间序列样本

        Args:
            signal_data: 输入信号数组 (channels, samples)
            labels: 标签数组 (samples,)

        Returns:
            (windows, window_labels) 元组
        """
        try:
            n_channels, n_samples = signal_data.shape

            # 计算窗口数量
            n_windows = (n_samples - self.window_size) // self.step_size + 1

            # 初始化窗口数组
            windows = np.zeros((n_windows, n_channels, self.window_size), dtype=np.float32)
            window_labels = np.zeros(n_windows, dtype=np.int32)

            # 创建滑动窗口
            for i in range(n_windows):
                start_idx = i * self.step_size
                end_idx = start_idx + self.window_size

                # 提取窗口数据
                windows[i] = signal_data[:, start_idx:end_idx]

                # 窗口标签采用窗口中间位置的标签
                mid_idx = start_idx + self.window_size // 2
                if mid_idx < len(labels):
                    window_labels[i] = labels[mid_idx]
                else:
                    window_labels[i] = labels[-1]

            logger.info(f"创建滑动窗口完成，窗口数量: {n_windows}, 窗口大小: {self.window_size}")
            return windows, window_labels

        except Exception as e:
            logger.error(f"创建滑动窗口失败: {str(e)}")
            return signal_data[np.newaxis, :, :], labels[np.newaxis]

    def extract_statistical_features(self, signal_data: np.ndarray) -> np.ndarray:
        """
        提取统计特征

        Args:
            signal_data: 输入信号数组 (channels, samples)

        Returns:
            特征数组
        """
        try:
            n_channels, n_samples = signal_data.shape
            features = []

            for channel in range(n_channels):
                signal_channel = signal_data[channel]

                # 时域特征
                mean = np.mean(signal_channel)
                std = np.std(signal_channel)
                rms = np.sqrt(np.mean(signal_channel ** 2))
                peak = np.max(np.abs(signal_channel))
                peak_to_peak = np.max(signal_channel) - np.min(signal_channel)
                skewness = self._calculate_skewness(signal_channel)
                kurtosis = self._calculate_kurtosis(signal_channel)

                # 频域特征
                fft_spectrum = np.fft.fft(signal_channel)
                magnitude = np.abs(fft_spectrum)
                power = magnitude ** 2

                spectral_centroid = np.sum(np.arange(len(power)) * power) / np.sum(power)
                spectral_bandwidth = np.sqrt(
                    np.sum(((np.arange(len(power)) - spectral_centroid) ** 2) * power) / np.sum(power))

                channel_features = [
                    mean, std, rms, peak, peak_to_peak, skewness, kurtosis,
                    spectral_centroid, spectral_bandwidth
                ]

                features.extend(channel_features)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"提取统计特征失败: {str(e)}")
            return np.array([])

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def remove_outliers(self, signal_data: np.ndarray,
                        threshold: float = 3.0) -> np.ndarray:
        """
        去除异常值

        Args:
            signal_data: 输入信号数组
            threshold: 异常值阈值（标准差倍数）

        Returns:
            去除异常值后的信号数组
        """
        try:
            cleaned_data = np.zeros_like(signal_data)

            for channel in range(signal_data.shape[0]):
                signal_channel = signal_data[channel]

                # 计算Z-score
                z_scores = np.abs(zscore(signal_channel))

                # 标记异常值
                outlier_mask = z_scores > threshold

                # 使用插值替换异常值
                if np.any(outlier_mask):
                    x = np.arange(len(signal_channel))
                    good_indices = ~outlier_mask

                    if np.any(good_indices):
                        cleaned_data[channel] = np.interp(
                            x, x[good_indices], signal_channel[good_indices]
                        )
                    else:
                        cleaned_data[channel] = signal_channel
                else:
                    cleaned_data[channel] = signal_channel

            logger.info(f"异常值去除完成，阈值: {threshold}")
            return cleaned_data.astype(np.float32)

        except Exception as e:
            logger.error(f"异常值去除失败: {str(e)}")
            return signal_data

    def apply_data_augmentation(self, signal_data: np.ndarray,
                                noise_level: float = 0.01,
                                scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """
        应用数据增强

        Args:
            signal_data: 输入信号数组
            noise_level: 噪声水平
            scale_range: 缩放范围

        Returns:
            增强后的信号数组
        """
        try:
            augmented_data = np.zeros_like(signal_data)

            for channel in range(signal_data.shape[0]):
                signal_channel = signal_data[channel]

                # 添加高斯噪声
                noise = np.random.normal(0, noise_level * np.std(signal_channel), signal_channel.shape)
                augmented_signal = signal_channel + noise

                # 幅度缩放
                scale_factor = np.random.uniform(scale_range[0], scale_range[1])
                augmented_signal = augmented_signal * scale_factor

                augmented_data[channel] = augmented_signal

            logger.info(f"数据增强完成，噪声水平: {noise_level}, 缩放范围: {scale_range}")
            return augmented_data.astype(np.float32)

        except Exception as e:
            logger.error(f"数据增强失败: {str(e)}")
            return signal_data

    def preprocess_pipeline(self, signal_data: np.ndarray,
                            labels: Optional[np.ndarray] = None,
                            apply_denoising: bool = True,
                            apply_filtering: bool = True,
                            apply_normalization: bool = True,
                            apply_outlier_removal: bool = True,
                            create_windows: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        完整的预处理流水线

        Args:
            signal_data: 输入信号数组
            labels: 标签数组
            apply_denoising: 是否应用去噪
            apply_filtering: 是否应用滤波
            apply_normalization: 是否应用归一化
            apply_outlier_removal: 是否去除异常值
            create_windows: 是否创建滑动窗口

        Returns:
            (processed_data, processed_labels) 元组
        """
        logger.info("开始预处理流水线")
        processed_data = signal_data.copy()

        try:
            # 1. 异常值去除
            if apply_outlier_removal:
                processed_data = self.remove_outliers(processed_data)
                logger.info("异常值去除完成")

            # 2. 信号去噪
            if apply_denoising:
                processed_data = self.wavelet_denoising(processed_data)
                logger.info("信号去噪完成")

            # 3. 信号滤波
            if apply_filtering:
                processed_data = self.butterworth_filter(processed_data, highcut=10000)
                logger.info("信号滤波完成")

            # 4. 信号归一化
            if apply_normalization:
                processed_data = self.normalize_signal(processed_data, method='minmax')
                logger.info("信号归一化完成")

            # 5. 创建滑动窗口
            if create_windows and labels is not None:
                window_data, window_labels = self.create_sliding_windows(processed_data, labels)
                # 👉👉👉 关键修复：转置数据形状 👈👈👈
                # 从 (n_windows, n_channels, window_size) 转换为 (n_windows, window_size, n_channels)
                window_data = np.transpose(window_data, (0, 2, 1))
                logger.info(f"滑动窗口创建完成，转置后形状: {window_data.shape}")
                return window_data, window_labels

            logger.info("预处理流水线完成")
            return processed_data, labels

        except Exception as e:
            logger.error(f"预处理流水线失败: {str(e)}")
            return signal_data, labels