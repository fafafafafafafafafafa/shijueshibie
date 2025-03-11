# utils/import_utils.py
"""
模块导入和路径处理工具

提供辅助函数，用于:
- 确保模块路径在系统路径中
- 动态导入模块
- 检测模块是否可用
"""
import sys
import os
import importlib
import logging


def ensure_module_path(path=None):
    """
    确保指定路径在Python搜索路径中

    Args:
        path: 要添加到Python路径的目录，None则使用当前文件所在目录

    Returns:
        bool: 是否添加了新路径
    """
    # 如果未指定路径，使用当前文件所在目录
    if path is None:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        current_dir = os.path.abspath(path)

    # 获取父目录（utils的上一级）
    parent_dir = os.path.dirname(current_dir)

    added = False

    # 检查并添加当前目录
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        added = True

    # 检查并添加父目录
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        added = True

    # 检查项目根目录（假设父目录的父目录是项目根目录）
    root_dir = os.path.dirname(parent_dir)
    if os.path.exists(root_dir) and root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        added = True

    return added


def import_module_safely(module_name):
    """
    安全地导入模块，返回模块对象或None（如果导入失败）

    Args:
        module_name: 要导入的模块名称

    Returns:
        module 或 None: 导入的模块或None（如果导入失败）
    """
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        print(f"导入模块 {module_name} 失败: {e}")
        return None


def import_module_from_path(module_path):
    """
    从指定路径导入模块

    Args:
        module_path: 模块路径字符串，如 'plugin_modules.yolo_detector'

    Returns:
        module 或 None: 导入的模块或None（如果导入失败）
    """
    try:
        # 确保模块路径在系统路径中
        ensure_module_path()

        # 尝试多种导入方式
        try:
            # 直接导入
            module = importlib.import_module(module_path)
        except ImportError:
            # 尝试相对导入
            if '.' in module_path:
                # 尝试从父包导入
                parent_package, module_name = module_path.rsplit('.', 1)
                parent = importlib.import_module(parent_package)
                module = getattr(parent, module_name)
            else:
                # 重新抛出异常
                raise

        print(f"成功导入模块: {module_path}")
        if hasattr(module, "__file__"):
            print(f"模块路径: {module.__file__}")
        return module
    except ImportError as e:
        print(f"导入错误: {e}")
        # 尝试在错误处理中解决路径问题
        try:
            # 尝试从utils包导入
            if not module_path.startswith('utils.'):
                alt_path = f'utils.{module_path}'
                print(f"尝试备用路径: {alt_path}")
                return importlib.import_module(alt_path)
        except ImportError:
            pass
        return None


def ensure_directories(*dirs):
    """
    确保指定的目录存在

    Args:
        *dirs: 要确保存在的目录列表

    Returns:
        list: 创建的目录列表
    """
    created = []
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                created.append(dir_path)
                print(f"已创建目录: {dir_path}")
            except Exception as e:
                print(f"创建目录 {dir_path} 时出错: {e}")
    return created


# 如果作为脚本执行，测试功能
if __name__ == "__main__":
    # 确保当前目录和必要的日志目录存在
    ensure_directories("logs", "logs/cache_stats")

    ensure_module_path()
    try:
        test_module = "plugin_modules.yolo_detector"
        module = import_module_safely(test_module)
        if module:
            print(f"模块路径: {module.__file__}")
    except Exception as e:
        print(f"测试导入时出错: {e}")
