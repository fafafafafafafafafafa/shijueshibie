# -*- coding: utf-8 -*-
"""
日志系统配置模块 - 提供完整的日志配置功能

本模块提供以下功能:
1. setup_logger - 创建并配置支持UTF-8的日志记录器
2. setup_utf8_console - 配置控制台输出为UTF-8编码
3. init_root_logger - 初始化根日志记录器，确保系统范围内的日志都支持UTF-8
4. get_logger - 获取或创建已配置的日志记录器
5. ensure_log_directories - 确保日志目录存在

使用示例:
    from logger_config import setup_logger, get_logger

    # 创建新的日志记录器
    logger = setup_logger("my_module")
    logger.info("这是一条日志")

    # 在其他地方获取同一日志记录器
    same_logger = get_logger("my_module")
"""

import logging
import sys
import io
import os
import threading

# 用于存储已配置的日志记录器的字典
_loggers = {}
_loggers_lock = threading.RLock()

# 默认日志目录
DEFAULT_LOG_DIR = "logs"
CACHE_STATS_DIR = "logs/cache_stats"


def ensure_log_directories():
    """
    确保所有日志所需的目录存在

    Returns:
        bool: 是否成功创建或确认目录存在
    """
    try:
        # 确保主日志目录存在
        if not os.path.exists(DEFAULT_LOG_DIR):
            os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

        # 确保缓存统计目录存在
        if not os.path.exists(CACHE_STATS_DIR):
            os.makedirs(CACHE_STATS_DIR, exist_ok=True)

        return True
    except Exception as e:
        print(f"创建日志目录时出错: {e}")
        return False


def setup_logger(name, log_file=None, level=logging.INFO, format_str=None):
    """
    创建并配置支持UTF-8的日志记录器

    如果具有相同名称的日志记录器已存在，则返回现有记录器

    Args:
        name: 日志记录器名称
        log_file: 可选的日志文件路径
        level: 日志级别
        format_str: 日志格式字符串

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    ensure_log_directories()

    # 检查是否已经创建了该名称的日志记录器
    with _loggers_lock:
        if name in _loggers:
            return _loggers[name]

    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有处理器
    logger.handlers.clear()

    # 默认格式
    if not format_str:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果指定了文件）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 存储日志记录器以便复用
    with _loggers_lock:
        _loggers[name] = logger

    return logger


def get_logger(name, default_level=logging.INFO, log_file=None):
    """
    获取现有日志记录器或创建新的日志记录器

    Args:
        name: 日志记录器名称
        default_level: 如果需要创建新日志记录器时使用的默认级别
        log_file: 可选的日志文件路径，仅在创建新记录器时使用

    Returns:
        logging.Logger: 请求的日志记录器
    """
    with _loggers_lock:
        if name in _loggers:
            return _loggers[name]

    # 如果不存在，创建新的
    if log_file is None and name != 'root':
        # 为非根日志记录器创建默认日志文件
        log_file = os.path.join(DEFAULT_LOG_DIR, f"{name}.log")

    return setup_logger(name, log_file=log_file, level=default_level)


def setup_utf8_console():
    """
    配置控制台输出为UTF-8编码

    Returns:
        bool: 设置是否成功
    """
    try:
        # Windows系统特别处理
        if sys.platform.startswith('win'):
            # 强制使用UTF-8输出
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

            # 在Windows命令行中可能还需要设置控制台代码页
            try:
                import subprocess
                subprocess.run(['chcp', '65001'], shell=True, check=False)
            except:
                pass

        return True
    except Exception as e:
        print(f"设置UTF-8控制台输出时出错: {e}")
        return False


def init_root_logger(log_file=None, level=logging.INFO):
    """
    初始化根日志记录器，确保系统范围内的日志都支持UTF-8

    Args:
        log_file: 可选的日志文件路径
        level: 日志级别

    Returns:
        logging.Logger: 配置好的根日志记录器
    """
    # 确保日志目录存在
    ensure_log_directories()

    # 首先设置控制台UTF-8支持
    setup_utf8_console()

    # 使用默认日志文件（如果未指定）
    if log_file is None:
        log_file = os.path.join(DEFAULT_LOG_DIR, "app.log")

    # 配置根日志记录器
    root_logger = setup_logger('root', log_file, level)

    # 存储根日志记录器以便复用
    with _loggers_lock:
        _loggers['root'] = root_logger

    return root_logger


def configure_module_logger(module_name, log_dir=DEFAULT_LOG_DIR,
                            level=logging.INFO):
    """
    为指定模块配置日志记录器，并自动设置合适的日志文件

    Args:
        module_name: 模块名称
        log_dir: 日志目录
        level: 日志级别

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件路径: logs/module_name.log
    log_file = os.path.join(log_dir, f"{module_name}.log")

    # 设置并返回日志记录器
    return setup_logger(module_name, log_file, level)


# 初始化模块时确保日志目录存在
ensure_log_directories()

# 如果直接运行此模块，执行测试
if __name__ == "__main__":
    # 测试日志系统
    print("测试日志系统...")

    # 初始化根日志记录器
    root_logger = init_root_logger()
    root_logger.info("根日志记录器测试")

    # 创建模块日志记录器
    test_logger = get_logger("test_module")
    test_logger.info("模块日志记录器测试")

    print(f"日志文件应该已创建在 {DEFAULT_LOG_DIR} 目录中")
