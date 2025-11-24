#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志系统模块
提供统一的日志记录功能
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

from ..config.config import LogConfig, LOGS_DIR

def setup_logger(
    log_level: str = LogConfig.LOG_LEVEL,
    log_file: Optional[Path] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logger:
    """
    设置和配置日志系统
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 日志文件路径，如果为None则使用默认路径
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        
    Returns:
        配置好的logger实例
    """
    
    # 移除默认的logger配置
    logger.remove()
    
    # 确保日志目录存在
    LOGS_DIR.mkdir(exist_ok=True)
    
    # 设置日志文件路径
    if log_file is None:
        log_file = LogConfig.LOG_FILE
    
    # 配置控制台输出
    if console_output:
        logger.add(
            sys.stdout,
            level=log_level,
            format=LogConfig.LOG_FORMAT,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # 配置文件输出
    if file_output:
        logger.add(
            log_file,
            level=log_level,
            format=LogConfig.LOG_FORMAT,
            rotation=LogConfig.LOG_ROTATION,
            retention=LogConfig.LOG_RETENTION,
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )
    
    return logger

def get_module_logger(module_name: str):
    """
    获取指定模块的logger
    
    Args:
        module_name: 模块名称
        
    Returns:
        模块logger实例
    """
    return logger.bind(module=module_name)

# 创建默认logger实例
logger = setup_logger()

# 导出logger
__all__ = ['logger', 'setup_logger', 'get_module_logger']