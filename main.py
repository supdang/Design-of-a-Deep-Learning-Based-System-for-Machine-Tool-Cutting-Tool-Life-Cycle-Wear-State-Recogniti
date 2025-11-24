#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于深度学习的机床刀具生命周期磨损状态识别与诊断系统
Main Application Entry Point

"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.gui.main_window import MainWindow
from src.utils.logger import setup_logger

def main():
    """主函数 - 应用程序入口点"""
    
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='刀具磨损状态识别与诊断系统')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='设置日志级别')
    args = parser.parse_args()
    
    # 设置日志系统
    logger = setup_logger(log_level=args.log_level)
    logger.info("启动刀具磨损状态识别与诊断系统")
    
    try:
        # 创建Qt应用程序
        app = QApplication(sys.argv)
        app.setApplicationName("刀具磨损诊断系统")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("齐鲁理工学院")
        
        # 设置应用程序样式
        app.setStyle('Fusion')
        
        # 创建主窗口
        main_window = MainWindow(debug_mode=args.debug)
        main_window.show()
        
        # 运行应用程序
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
