#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主窗口模块
实现系统的主界面和核心功能
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QMessageBox, QProgressBar, QGroupBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QSplitter,
    QFrame, QStatusBar, QAction, QMenuBar, QToolBar, QRadioButton,
    QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import torch

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun', 'Arial Unicode MS']  # 尝试使用系统中存在的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

from ..utils.logger import get_module_logger
from ..config.config import GUIConfig, DataConfig, ModelConfig
from ..data.data_loader import PHM2010DataLoader
from ..data.preprocessor import SignalPreprocessor
from ..models.lstm_model import ToolWearClassifier, ToolWearDataset, DataLoader, LSTMToolWearModel

logger = get_module_logger(__name__)


class TrainingWorker(QThread):
    """训练工作线程"""

    progress_updated = pyqtSignal(int, float, float, float, float)
    training_finished = pyqtSignal(dict)
    training_error = pyqtSignal(str)

    def __init__(self, classifier, train_data, val_data, config):
        super().__init__()
        self.classifier = classifier
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self._is_running = True

    def run(self):
        """执行训练"""
        try:
            # 设置训练参数
            batch_size = self.config.get('batch_size', ModelConfig.BATCH_SIZE)
            num_epochs = self.config.get('num_epochs', ModelConfig.NUM_EPOCHS)
            learning_rate = self.config.get('learning_rate', ModelConfig.LEARNING_RATE)
            patience = self.config.get('patience', ModelConfig.PATIENCE)

            # 检查设备类型
            device_type = str(self.classifier.device)
            logger.info(f"使用设备: {device_type}")

            if 'cuda' in device_type:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                logger.info(f"检测到GPU内存: {gpu_mem:.2f} GB")

                # 根据GPU内存自动调整batch size
                if gpu_mem < 4.5:
                    batch_size = min(batch_size, 4)
                    logger.warning(f"GPU内存严重不足，自动减小batch size到: {batch_size}")
                elif gpu_mem < 6:
                    batch_size = min(batch_size, 8)
                    logger.warning(f"GPU内存不足，自动减小batch size到: {batch_size}")

            # 创建数据加载器
            train_dataset = ToolWearDataset(self.train_data[0], self.train_data[1])
            val_dataset = ToolWearDataset(self.val_data[0], self.val_data[1])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 清理内存
            if 'cuda' in device_type:
                torch.cuda.empty_cache()

            # 训练循环
            for epoch in range(num_epochs):
                if not self._is_running:
                    break

                # 训练一个epoch
                train_loss, train_acc = self._train_epoch(train_loader)
                val_loss, val_acc = self._validate(val_loader)

                # 每个epoch后清理缓存
                if 'cuda' in device_type:
                    torch.cuda.empty_cache()

                # 发送进度信号
                self.progress_updated.emit(
                    epoch + 1, train_loss, train_acc, val_loss, val_acc
                )

                # 检查早停
                if epoch > patience and val_acc < 0.5:
                    break

            # 评估模型
            results = self.classifier.evaluate(self.val_data)
            self.training_finished.emit(results)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                error_msg = "GPU内存严重不足！尝试以下方法：\n1. 减小批量大小（建议≤4）\n2. 减小模型大小（隐藏层≤32）\n3. 关闭其他GPU程序\n4. 切换到CPU模式"
                self.training_error.emit(error_msg)
            else:
                self.training_error.emit(str(e))
        except Exception as e:
            self.training_error.emit(str(e))

    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        self.classifier.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.classifier.device), target.to(self.classifier.device)

            self.classifier.optimizer.zero_grad()
            output = self.classifier.model(data)
            loss = self.classifier.criterion(output, target)
            loss.backward()
            self.classifier.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

            # 每个batch后清理缓存（GPU专用）
            if batch_idx % 4 == 0 and 'cuda' in str(self.classifier.device):
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def _validate(self, val_loader):
        """验证模型"""
        self.classifier.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.classifier.device), target.to(self.classifier.device)

                output = self.classifier.model(data)
                loss = self.classifier.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def stop(self):
        """停止训练"""
        self._is_running = False
        # 停止时清理内存
        if 'cuda' in str(self.classifier.device):
            torch.cuda.empty_cache()


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode

        # 初始化数据
        self.current_data = None
        self.current_labels = None
        self.classifier = None
        self.training_worker = None

        # 初始化UI
        self.init_ui()
        self.setup_menu_bar()
        self.setup_tool_bar()
        self.setup_status_bar()

        # 设置样式
        self.setup_styles()

        logger.info("主窗口初始化完成")

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle(GUIConfig.WINDOW_TITLE)
        self.setGeometry(100, 100, GUIConfig.WINDOW_WIDTH, GUIConfig.WINDOW_HEIGHT)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # 创建右侧面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)

        # 设置布局比例
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)

    def create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 数据管理组
        data_group = self.create_data_group()
        layout.addWidget(data_group)

        # 设备选择组
        device_group = self.create_device_group()
        layout.addWidget(device_group)

        # 模型配置组
        model_group = self.create_model_group()
        layout.addWidget(model_group)

        # 训练控制组
        training_group = self.create_training_group()
        layout.addWidget(training_group)

        # 预测控制组
        prediction_group = self.create_prediction_group()
        layout.addWidget(prediction_group)

        # 添加弹簧
        layout.addStretch()

        return panel

    def create_device_group(self) -> QGroupBox:
        """创建设备选择组"""
        group = QGroupBox("计算设备")
        layout = QVBoxLayout(group)

        # CPU/GPU 选择
        device_layout = QHBoxLayout()

        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU")

        # 检查GPU可用性
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.gpu_radio.setText(f"GPU (可用内存: {gpu_mem:.1f}GB)")
            self.gpu_radio.setChecked(True)
            self.gpu_radio.setEnabled(True)
        else:
            self.gpu_radio.setText("GPU (不可用)")
            self.gpu_radio.setEnabled(False)
            self.cpu_radio.setChecked(True)

        # 创建按钮组
        self.device_group = QButtonGroup()
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.gpu_radio)

        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)
        device_layout.addStretch()

        layout.addLayout(device_layout)

        # 设备信息
        self.device_info_label = QLabel("")
        layout.addWidget(self.device_info_label)

        # 更新设备信息
        self.update_device_info()

        return group

    def update_device_info(self):
        """更新设备信息"""
        if self.cpu_radio.isChecked():
            self.device_info_label.setText("使用CPU进行计算，速度较慢但内存无限制")
            self.device_info_label.setStyleSheet("color: #EBCB8B;")
        else:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if gpu_mem < 4.5:
                    self.device_info_label.setText("⚠️ GPU内存严重不足，建议使用CPU或减小模型参数")
                    self.device_info_label.setStyleSheet("color: #BF616A;")
                elif gpu_mem < 6:
                    self.device_info_label.setText("⚠️ GPU内存不足，建议减小批量大小和模型大小")
                    self.device_info_label.setStyleSheet("color: #EBCB8B;")
                else:
                    self.device_info_label.setText("GPU内存充足，可使用较大批量和模型")
                    self.device_info_label.setStyleSheet("color: #A3BE8C;")
            else:
                self.device_info_label.setText("GPU不可用，请使用CPU")
                self.device_info_label.setStyleSheet("color: #BF616A;")

    def create_data_group(self) -> QGroupBox:
        """创建数据管理组"""
        group = QGroupBox("数据管理")
        layout = QVBoxLayout(group)

        # 数据路径
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("数据路径:"))
        self.data_path_edit = QLineEdit()
        path_layout.addWidget(self.data_path_edit)

        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_data_path)
        path_layout.addWidget(browse_btn)

        layout.addLayout(path_layout)

        # 数据加载按钮
        load_btn = QPushButton("加载数据")
        load_btn.clicked.connect(self.load_data)
        layout.addWidget(load_btn)

        # 数据信息
        self.data_info_label = QLabel("未加载数据")
        layout.addWidget(self.data_info_label)

        return group

    def create_model_group(self) -> QGroupBox:
        """创建模型配置组"""
        group = QGroupBox("模型配置")
        layout = QVBoxLayout(group)

        # 隐藏层大小
        hidden_layout = QHBoxLayout()
        hidden_layout.addWidget(QLabel("隐藏层大小:"))
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(8, 128)
        self.hidden_size_spin.setValue(32)
        hidden_layout.addWidget(self.hidden_size_spin)
        layout.addLayout(hidden_layout)

        # LSTM层数
        layers_layout = QHBoxLayout()
        layers_layout.addWidget(QLabel("LSTM层数:"))
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 1)
        self.num_layers_spin.setValue(1)
        self.num_layers_spin.setEnabled(False)
        layers_layout.addWidget(self.num_layers_spin)
        layout.addLayout(layers_layout)

        # Dropout率
        dropout_layout = QHBoxLayout()
        dropout_layout.addWidget(QLabel("Dropout率:"))
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.5)
        self.dropout_spin.setSingleStep(0.1)
        self.dropout_spin.setValue(0.2)
        dropout_layout.addWidget(self.dropout_spin)
        layout.addLayout(dropout_layout)

        return group

    def create_training_group(self) -> QGroupBox:
        """创建训练控制组"""
        group = QGroupBox("训练控制")
        layout = QVBoxLayout(group)

        # 批量大小
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("批量大小:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(2, 8)
        self.batch_size_spin.setValue(4)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)

        # 训练轮数
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("训练轮数:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 100)
        self.epochs_spin.setValue(30)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)

        # 学习率
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学习率:"))
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.01)
        self.learning_rate_spin.setSingleStep(0.001)
        self.learning_rate_spin.setValue(0.001)
        lr_layout.addWidget(self.learning_rate_spin)
        layout.addLayout(lr_layout)

        # 训练按钮
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)

        # 停止按钮
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        # 训练进度
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        return group

    def create_prediction_group(self) -> QGroupBox:
        """创建预测控制组"""
        group = QGroupBox("预测控制")
        layout = QVBoxLayout(group)

        # 预测按钮
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        # 保存模型按钮
        self.save_model_btn = QPushButton("保存模型")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        layout.addWidget(self.save_model_btn)

        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_btn)

        return group

    def create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 创建标签页
        self.tab_widget = QTabWidget()

        # 数据可视化标签页
        self.data_tab = self.create_data_tab()
        self.tab_widget.addTab(self.data_tab, "数据可视化")

        # 训练监控标签页
        self.training_tab = self.create_training_tab()
        self.tab_widget.addTab(self.training_tab, "训练监控")

        # 结果分析标签页
        self.results_tab = self.create_results_tab()
        self.tab_widget.addTab(self.results_tab, "结果分析")

        layout.addWidget(self.tab_widget)

        return panel

    def create_data_tab(self) -> QWidget:
        """创建数据可视化标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建matplotlib图形
        self.data_figure, self.data_axes = plt.subplots(2, 2, figsize=(10, 8))
        self.data_canvas = FigureCanvas(self.data_figure)

        # 添加工具栏
        toolbar = NavigationToolbar(self.data_canvas, tab)
        layout.addWidget(toolbar)
        layout.addWidget(self.data_canvas)

        return tab

    def create_training_tab(self) -> QWidget:
        """创建训练监控标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建matplotlib图形
        self.training_figure, self.training_axes = plt.subplots(1, 2, figsize=(12, 5))
        self.training_canvas = FigureCanvas(self.training_figure)

        # 添加工具栏
        toolbar = NavigationToolbar(self.training_canvas, tab)
        layout.addWidget(toolbar)
        layout.addWidget(self.training_canvas)

        return tab

    def create_results_tab(self) -> QWidget:
        """创建结果分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建matplotlib图形
        self.results_figure, self.results_axes = plt.subplots(2, 2, figsize=(12, 10))
        self.results_canvas = FigureCanvas(self.results_figure)

        # 添加工具栏
        toolbar = NavigationToolbar(self.results_canvas, tab)
        layout.addWidget(toolbar)
        layout.addWidget(self.results_canvas)

        return tab

    def setup_menu_bar(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # 新建动作
        new_action = QAction('新建', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        # 打开动作
        open_action = QAction('打开', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        # 保存动作
        save_action = QAction('保存', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 工具菜单
        tool_menu = menubar.addMenu('工具')

        # 设置动作
        settings_action = QAction('设置', self)
        settings_action.triggered.connect(self.show_settings)
        tool_menu.addAction(settings_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        # 关于动作
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_tool_bar(self):
        """设置工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # 新建按钮
        new_btn = QPushButton("新建")
        new_btn.clicked.connect(self.new_project)
        toolbar.addWidget(new_btn)

        # 打开按钮
        open_btn = QPushButton("打开")
        open_btn.clicked.connect(self.open_project)
        toolbar.addWidget(open_btn)

        # 保存按钮
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_project)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        # 训练按钮
        train_btn = QPushButton("训练")
        train_btn.clicked.connect(self.start_training)
        toolbar.addWidget(train_btn)

        # 预测按钮
        predict_btn = QPushButton("预测")
        predict_btn.clicked.connect(self.start_prediction)
        toolbar.addWidget(predict_btn)

    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def setup_styles(self):
        """设置样式"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {GUIConfig.PRIMARY_COLOR};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {GUIConfig.SECONDARY_COLOR};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QPushButton {{
                background-color: {GUIConfig.ACCENT_COLOR};
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #4C566A;
            }}
            QPushButton:pressed {{
                background-color: #2E3440;
            }}
            QPushButton:disabled {{
                background-color: #D8DEE9;
                color: #4C566A;
            }}
            QLabel.warning {{
                color: #BF616A;
                font-weight: bold;
            }}
            QLabel.info {{
                color: #88C0D0;
            }}
        """)

    def browse_data_path(self):
        """浏览数据路径"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "",
            "MAT Files (*.mat);;CSV Files (*.csv);;All Files (*.*)"
        )
        if file_path:
            self.data_path_edit.setText(file_path)

    def load_data(self):
        """加载数据"""
        try:
            data_path = self.data_path_edit.text()
            if not data_path:
                QMessageBox.warning(self, "警告", "请选择数据文件")
                return

            data_dir = Path(data_path).parent
            data_loader = PHM2010DataLoader(data_dir)

            # 加载数据
            sensor_data, labels, file_names = data_loader.load_dataset()

            if sensor_data is None or labels is None:
                QMessageBox.warning(self, "警告", "数据加载失败")
                return

            self.current_data = sensor_data
            self.current_labels = labels

            # 更新数据信息
            info_text = f"数据形状: {sensor_data.shape}\n标签分布: {np.bincount(labels)}"
            self.data_info_label.setText(info_text)

            # 可视化数据
            self.visualize_data(sensor_data, labels)

            # 启用训练按钮
            self.train_btn.setEnabled(True)

            self.status_bar.showMessage("数据加载完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")
            logger.error(f"数据加载失败: {str(e)}")

    def visualize_data(self, sensor_data: np.ndarray, labels: np.ndarray):
        """可视化数据"""
        try:
            self.data_axes[0, 0].clear()
            self.data_axes[0, 1].clear()
            self.data_axes[1, 0].clear()
            self.data_axes[1, 1].clear()

            # 绘制时域信号
            time_axis = np.arange(sensor_data.shape[1]) / DataConfig.SAMPLE_RATE
            for i in range(min(3, sensor_data.shape[0])):
                self.data_axes[0, 0].plot(time_axis[:1000], sensor_data[i, :1000],
                                          label=f'通道 {i + 1}', alpha=0.7)

            self.data_axes[0, 0].set_xlabel('时间 (s)')
            self.data_axes[0, 0].set_ylabel('幅值')
            self.data_axes[0, 0].set_title('时域信号')
            self.data_axes[0, 0].legend()
            self.data_axes[0, 0].grid(True)

            # 绘制频域信号
            freq_axis = np.fft.fftfreq(sensor_data.shape[1], 1 / DataConfig.SAMPLE_RATE)
            fft_data = np.abs(np.fft.fft(sensor_data[0]))

            # 只显示正频率部分
            positive_freq_idx = freq_axis > 0
            self.data_axes[0, 1].plot(freq_axis[positive_freq_idx],
                                      fft_data[positive_freq_idx])
            self.data_axes[0, 1].set_xlabel('频率 (Hz)')
            self.data_axes[0, 1].set_ylabel('幅值')
            self.data_axes[0, 1].set_title('频域信号')
            self.data_axes[0, 1].grid(True)

            # 绘制标签分布
            label_counts = np.bincount(labels)
            label_names = ['初期磨损', '正常磨损', '后期磨损', '失效状态']

            self.data_axes[1, 0].bar(label_names[:len(label_counts)], label_counts)
            self.data_axes[1, 0].set_xlabel('磨损状态')
            self.data_axes[1, 0].set_ylabel('样本数量')
            self.data_axes[1, 0].set_title('标签分布')
            self.data_axes[1, 0].tick_params(axis='x', rotation=45)

            # 绘制信号统计特征
            features = []
            for i in range(min(sensor_data.shape[0], 4)):
                signal = sensor_data[i]
                features.extend([
                    np.mean(signal),
                    np.std(signal),
                    np.max(np.abs(signal)),
                    np.sqrt(np.mean(signal ** 2))
                ])

            feature_names = ['均值', '标准差', '峰值', 'RMS'] * min(sensor_data.shape[0], 4)

            self.data_axes[1, 1].bar(range(len(features)), features)
            self.data_axes[1, 1].set_xlabel('特征')
            self.data_axes[1, 1].set_ylabel('数值')
            self.data_axes[1, 1].set_title('统计特征')
            self.data_axes[1, 1].set_xticks(range(len(feature_names)))
            self.data_axes[1, 1].set_xticklabels(feature_names, rotation=45)

            self.data_canvas.draw()

        except Exception as e:
            logger.error(f"数据可视化失败: {str(e)}")

    def create_model(self, config):
        """创建模型 - 修正输入尺寸"""
        # 关键修复：input_size 固定为7（传感器通道数），不从配置中获取
        corrected_config = {
            'input_size': 7,  # 固定为7个传感器通道
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers'],
            'num_classes': config['num_classes'],
            'dropout_rate': config['dropout_rate']
        }

        return LSTMToolWearModel(**corrected_config)

    def start_training(self):
        """开始训练 - 修正版"""
        try:
            if self.current_data is None or self.current_labels is None:
                QMessageBox.warning(self, "警告", "请先加载数据")
                return

            # 获取设备选择
            if self.cpu_radio.isChecked():
                device = 'cpu'
                logger.info("选择使用CPU进行训练")
            else:
                if torch.cuda.is_available():
                    device = 'cuda'
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    logger.info(f"选择使用GPU进行训练，可用内存: {gpu_mem:.2f} GB")

                    # 4GB GPU专用配置
                    if gpu_mem < 4.5:
                        # 强制设置为4GB GPU安全参数
                        self.batch_size_spin.setValue(4)
                        self.hidden_size_spin.setValue(32)
                        self.epochs_spin.setValue(30)

                        reply = QMessageBox.warning(
                            self, "GPU内存严重不足",
                            f"检测到GPU内存非常小 ({gpu_mem:.2f} GB)，已自动设置安全参数：\n"
                            f"• 批量大小: 4\n"
                            f"• 隐藏层大小: 32\n"
                            f"• 训练轮数: 30\n"
                            f"• LSTM层数: 1\n\n"
                            "请确保关闭所有其他GPU程序（如浏览器、视频播放器等）。\n"
                            "是否继续训练？",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes
                        )
                        if reply == QMessageBox.No:
                            return
                else:
                    QMessageBox.warning(self, "GPU不可用", "CUDA不可用，将自动切换到CPU模式")
                    device = 'cpu'
                    self.cpu_radio.setChecked(True)

            # 数据预处理
            preprocessor = SignalPreprocessor()
            processed_data, processed_labels = preprocessor.preprocess_pipeline(
                self.current_data, self.current_labels
            )

            # 关键修复：确保数据形状为 (samples, timesteps, features)
            if processed_data.ndim == 2:
                # 重塑为 (samples, timesteps, features)
                processed_data = processed_data.reshape(
                    processed_data.shape[0],
                    DataConfig.WINDOW_SIZE,
                    len(DataConfig.SENSOR_CHANNELS)
                )
                logger.info(f"数据已重塑为形状: {processed_data.shape}")
            elif processed_data.ndim == 3:
                # 确保形状为 (samples, timesteps, features)
                if processed_data.shape[1] != DataConfig.WINDOW_SIZE or processed_data.shape[2] != 7:
                    processed_data = processed_data.reshape(
                        processed_data.shape[0],
                        DataConfig.WINDOW_SIZE,
                        len(DataConfig.SENSOR_CHANNELS)
                    )
                    logger.info(f"数据已重塑为形状: {processed_data.shape}")

            # 分割数据
            from sklearn.model_selection import train_test_split
            train_data, val_data, train_labels, val_labels = train_test_split(
                processed_data, processed_labels, test_size=0.2, random_state=42
            )

            # 创建模型配置（不包含 input_size）
            model_config = {
                'hidden_size': self.hidden_size_spin.value(),
                'num_layers': self.num_layers_spin.value(),
                'num_classes': len(np.unique(processed_labels)),
                'dropout_rate': self.dropout_spin.value()
            }

            self.classifier = ToolWearClassifier(
                model=self.create_model(model_config),
                device=device  # 关键：传入用户选择的设备
            )

            # 关键修复：初始化优化器
            self.classifier.setup_optimizer(
                learning_rate=self.learning_rate_spin.value(),
                weight_decay=ModelConfig.WEIGHT_DECAY
            )

            # 设置训练配置
            training_config = {
                'batch_size': self.batch_size_spin.value(),
                'num_epochs': self.epochs_spin.value(),
                'learning_rate': self.learning_rate_spin.value(),
                'patience': ModelConfig.PATIENCE
            }

            # 启动训练线程
            self.training_worker = TrainingWorker(
                self.classifier,
                (train_data, train_labels),
                (val_data, val_labels),
                training_config
            )

            self.training_worker.progress_updated.connect(self.update_training_progress)
            self.training_worker.training_finished.connect(self.training_completed)
            self.training_worker.training_error.connect(self.training_error)

            self.training_worker.start()

            # 更新UI状态
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setValue(0)

            device_name = "CPU" if device == 'cpu' else "GPU"
            self.status_bar.showMessage(f"正在使用{device_name}训练模型...")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练启动失败: {str(e)}")
            logger.error(f"训练启动失败: {str(e)}")

    def update_training_progress(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """更新训练进度"""
        try:
            # 更新进度条
            progress = (epoch / self.epochs_spin.value()) * 100
            self.progress_bar.setValue(int(progress))

            # 更新训练图表
            self.training_axes[0].clear()
            self.training_axes[1].clear()

            # 绘制损失曲线
            if hasattr(self.classifier, 'train_history'):
                history = self.classifier.train_history
                epochs = range(1, len(history['train_loss']) + 1)

                self.training_axes[0].plot(epochs, history['train_loss'], 'b-', label='训练损失')
                self.training_axes[0].plot(epochs, history['val_loss'], 'r-', label='验证损失')
                self.training_axes[0].set_xlabel('Epoch')
                self.training_axes[0].set_ylabel('Loss')
                self.training_axes[0].set_title('损失曲线')
                self.training_axes[0].legend()
                self.training_axes[0].grid(True)

                # 绘制准确率曲线
                self.training_axes[1].plot(epochs, history['train_acc'], 'b-', label='训练准确率')
                self.training_axes[1].plot(epochs, history['val_acc'], 'r-', label='验证准确率')
                self.training_axes[1].set_xlabel('Epoch')
                self.training_axes[1].set_ylabel('Accuracy')
                self.training_axes[1].set_title('准确率曲线')
                self.training_axes[1].legend()
                self.training_axes[1].grid(True)

            self.training_canvas.draw()

            # 更新状态栏
            self.status_bar.showMessage(f"训练中... Epoch {epoch}/{self.epochs_spin.value()}, "
                                        f"训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")

        except Exception as e:
            logger.error(f"更新训练进度失败: {str(e)}")

    def training_completed(self, results):
        """训练完成"""
        try:
            # 更新UI状态
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.predict_btn.setEnabled(True)
            self.save_model_btn.setEnabled(True)

            self.progress_bar.setValue(100)

            # 显示结果
            self.show_training_results(results)

            self.status_bar.showMessage("训练完成")

            QMessageBox.information(self, "成功", "模型训练完成！")

        except Exception as e:
            logger.error(f"训练完成处理失败: {str(e)}")

    def training_error(self, error_msg):
        """训练错误"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "训练错误", f"训练过程中出现错误: {error_msg}")
        logger.error(f"训练错误: {error_msg}")

    def stop_training(self):
        """停止训练"""
        if self.training_worker:
            self.training_worker.stop()
            self.training_worker.wait()

        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.status_bar.showMessage("训练已停止")

    def show_training_results(self, results):
        """显示训练结果"""
        try:
            self.results_axes[0, 0].clear()
            self.results_axes[0, 1].clear()
            self.results_axes[1, 0].clear()
            self.results_axes[1, 1].clear()

            # 绘制混淆矩阵
            conf_matrix = results['confusion_matrix']

            im = self.results_axes[0, 0].imshow(conf_matrix, interpolation='nearest', cmap='Blues')

            # 添加数值标签
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    self.results_axes[0, 0].text(j, i, format(conf_matrix[i, j], 'd'),
                                                 ha="center", va="center",
                                                 color="white" if conf_matrix[i, j] > thresh else "black")

            self.results_axes[0, 0].set_title('混淆矩阵')
            self.results_axes[0, 0].set_xlabel('预测标签')
            self.results_axes[0, 0].set_ylabel('真实标签')

            # 绘制分类报告
            class_report = results['classification_report']
            classes = ['初期磨损', '正常磨损', '后期磨损', '失效状态']
            metrics = ['precision', 'recall', 'f1-score']

            data = []
            for cls in classes:
                if cls in class_report:
                    row = [class_report[cls][metric] for metric in metrics]
                    data.append(row)

            if data:
                im = self.results_axes[0, 1].imshow(data, interpolation='nearest', cmap='RdYlBu')

                # 添加数值标签
                for i in range(len(data)):
                    for j in range(len(metrics)):
                        self.results_axes[0, 1].text(j, i, format(data[i][j], '.3f'),
                                                     ha="center", va="center")

                self.results_axes[0, 1].set_title('分类性能')
                self.results_axes[0, 1].set_xlabel('Metrics')
                self.results_axes[0, 1].set_ylabel('Classes')
                self.results_axes[0, 1].set_xticks(range(len(metrics)))
                self.results_axes[0, 1].set_xticklabels(metrics)
                self.results_axes[0, 1].set_yticks(range(len(classes)))
                self.results_axes[0, 1].set_yticklabels(classes)

            # 绘制预测概率分布
            probabilities = results['probabilities']
            predictions = results['predictions']

            # 为每个类别绘制概率分布
            for class_id in range(len(classes)):
                class_probs = probabilities[predictions == class_id][:, class_id]
                if len(class_probs) > 0:
                    self.results_axes[1, 0].hist(class_probs, alpha=0.7,
                                                 label=f'{classes[class_id]}', bins=20)

            self.results_axes[1, 0].set_xlabel('预测概率')
            self.results_axes[1, 0].set_ylabel('频次')
            self.results_axes[1, 0].set_title('预测概率分布')
            self.results_axes[1, 0].legend()

            # 绘制准确率指标
            accuracy = results['accuracy']

            # 创建准确率和其他指标的条形图
            metrics_values = [accuracy]
            metrics_names = ['总体准确率']

            # 添加各类别的F1分数
            for cls in classes:
                if cls in class_report and 'f1-score' in class_report[cls]:
                    metrics_values.append(class_report[cls]['f1-score'])
                    metrics_names.append(f'{cls} F1')

            bars = self.results_axes[1, 1].bar(metrics_names, metrics_values)
            self.results_axes[1, 1].set_ylabel('Score')
            self.results_axes[1, 1].set_title('性能指标')
            self.results_axes[1, 1].set_ylim(0, 1)

            # 为条形图添加数值标签
            for bar, value in zip(bars, metrics_values):
                self.results_axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                                             bar.get_height() + 0.01,
                                             f'{value:.3f}', ha='center', va='bottom')

            self.results_axes[1, 1].tick_params(axis='x', rotation=45)

            self.results_canvas.draw()

        except Exception as e:
            logger.error(f"显示训练结果失败: {str(e)}")

    def start_prediction(self):
        """开始预测"""
        try:
            if self.classifier is None or self.current_data is None:
                QMessageBox.warning(self, "警告", "请先训练模型并加载数据")
                return

            # 数据预处理
            preprocessor = SignalPreprocessor()
            processed_data, _ = preprocessor.preprocess_pipeline(
                self.current_data, create_windows=False
            )

            # 确保数据形状正确
            if processed_data.ndim == 2:
                processed_data = processed_data.reshape(
                    processed_data.shape[0],
                    DataConfig.WINDOW_SIZE,
                    len(DataConfig.SENSOR_CHANNELS)
                )

            # 预测
            predictions, probabilities = self.classifier.predict(processed_data)

            # 显示预测结果
            self.show_prediction_results(predictions, probabilities)

            self.status_bar.showMessage("预测完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")
            logger.error(f"预测失败: {str(e)}")

    def show_prediction_results(self, predictions, probabilities):
        """显示预测结果"""
        try:
            # 这里可以添加预测结果的可视化
            logger.info(f"预测完成，预测结果形状: {predictions.shape}")

        except Exception as e:
            logger.error(f"显示预测结果失败: {str(e)}")

    def save_model(self):
        """保存模型"""
        try:
            if self.classifier is None:
                QMessageBox.warning(self, "警告", "没有可保存的模型")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存模型", "", "PyTorch Models (*.pth)"
            )

            if file_path:
                self.classifier.save_model(file_path)
                QMessageBox.information(self, "成功", "模型保存成功！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型保存失败: {str(e)}")
            logger.error(f"模型保存失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "加载模型", "", "PyTorch Models (*.pth)"
            )

            if file_path:
                if self.classifier is None:
                    # 获取设备选择
                    device = 'cpu' if self.cpu_radio.isChecked() else 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.classifier = ToolWearClassifier(device=device)

                self.classifier.load_model(file_path)
                self.predict_btn.setEnabled(True)
                self.save_model_btn.setEnabled(True)

                QMessageBox.information(self, "成功", "模型加载成功！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            logger.error(f"模型加载失败: {str(e)}")

    def new_project(self):
        """新建项目"""
        reply = QMessageBox.question(self, "新建项目", "是否创建新项目？这将清除当前数据。")
        if reply == QMessageBox.Yes:
            self.reset_ui()

    def open_project(self):
        """打开项目"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "Project Files (*.proj)"
        )

        if file_path:
            try:
                # 这里可以实现项目文件的加载逻辑
                logger.info(f"打开项目: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"项目打开失败: {str(e)}")

    def save_project(self):
        """保存项目"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "Project Files (*.proj)"
        )

        if file_path:
            try:
                # 这里可以实现项目文件的保存逻辑
                logger.info(f"保存项目: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"项目保存失败: {str(e)}")

    def reset_ui(self):
        """重置UI"""
        self.current_data = None
        self.current_labels = None
        self.classifier = None

        self.data_path_edit.clear()
        self.data_info_label.setText("未加载数据")

        self.train_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)

        self.progress_bar.setValue(0)

        # 清空图表
        for ax in self.data_axes.flat:
            ax.clear()
        self.data_canvas.draw()

        for ax in self.training_axes.flat:
            ax.clear()
        self.training_canvas.draw()

        for ax in self.results_axes.flat:
            ax.clear()
        self.results_canvas.draw()

        self.status_bar.showMessage("就绪")

    def show_settings(self):
        """显示设置对话框"""
        QMessageBox.information(self, "设置", "设置功能待实现")

    def show_about(self):
        """显示关于对话框"""
        about_text = f"""
        <h2>{GUIConfig.WINDOW_TITLE}</h2>
        <p><b>版本:</b> 1.0.0</p>
        <p><b>作者:</b> 张述瑞</p>
        <p><b>学号:</b> 221171020211</p>
        <p><b>专业:</b> 2022级数据科学与大数据技术02班</p>
        <p><b>指导教师:</b> 田晓艺</p>
        <p><b>学校:</b> 齐鲁理工学院</p>
        <hr>
        <p>本系统基于深度学习技术，实现机床刀具生命周期磨损状态的智能识别与诊断。</p>
        """

        QMessageBox.about(self, "关于", about_text)

    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(self, "退出", "确定要退出程序吗？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 停止训练线程
            if self.training_worker and self.training_worker.isRunning():
                self.training_worker.stop()
                self.training_worker.wait()

            # 退出前清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            event.accept()
        else:
            event.ignore()