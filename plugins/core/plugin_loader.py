# -*- coding: utf-8 -*-
"""
插件加载器 - 负责发现和加载插件
"""

import os
import sys
import inspect
import importlib
import pkgutil
import logging
from typing import Dict, List, Any, Optional, Type, Set, Tuple

from .plugin_interface import PluginBase, PluginInfo, PluginType
from .plugin_registry import register_plugin

logger = logging.getLogger("PluginLoader")


def discover_plugins(plugin_dirs: Optional[List[str]] = None) -> Dict[
    str, List[Type[PluginBase]]]:
    """
    发现可用插件

    Args:
        plugin_dirs: 插件目录列表，None使用默认目录

    Returns:
        Dict[str, List[Type[PluginBase]]]: 插件类型到插件类的映射
    """
    result = {}

    # 如果未提供目录，使用默认目录
    if plugin_dirs is None:
        # 获取当前模块所在目录的上级目录
        current_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        plugin_dirs = [
            os.path.join(current_dir, "detectors"),
            os.path.join(current_dir, "recognizers"),
            os.path.join(current_dir, "mappers"),
            os.path.join(current_dir, "visualizers"),
            os.path.join(current_dir, "tools"),
        ]

        # 添加用户插件目录
        user_plugin_dir = os.path.expanduser("~/.pose_tracking/plugins")
        if os.path.exists(user_plugin_dir):
            plugin_dirs.append(user_plugin_dir)

    # 遍历每个目录
    for plugin_dir in plugin_dirs:
        if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
            logger.warning(f"插件目录不存在或不是目录: {plugin_dir}")
            continue

        # 将目录添加到Python路径
        if plugin_dir not in sys.path:
            sys.path.append(plugin_dir)

        # 遍历目录中的Python文件
        for filename in os.listdir(plugin_dir):
            if filename.endswith("_plugin.py") or filename.endswith(
                    "_Plugin.py"):
                module_name = filename[:-3]  # 去除.py后缀

                try:
                    # 动态导入模块
                    spec = importlib.util.spec_from_file_location(
                        module_name, os.path.join(plugin_dir, filename)
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # 在模块中查找插件类
                    plugins = _find_plugin_classes(module)

                    # 按类型分组
                    for plugin_class in plugins:
                        try:
                            # 创建临时实例以获取信息
                            temp_instance = plugin_class()
                            plugin_info = temp_instance.get_info()
                            plugin_type = plugin_info.type.value

                            if plugin_type not in result:
                                result[plugin_type] = []
                            result[plugin_type].append(plugin_class)

                            logger.debug(
                                f"发现插件: {plugin_info.id}, 类型: {plugin_type}")
                        except Exception as e:
                            logger.error(
                                f"获取插件信息时出错: {plugin_class.__name__}, 错误: {e}")
                except Exception as e:
                    logger.error(
                        f"加载插件模块时出错: {module_name}, 错误: {e}")

    # 记录发现结果
    for plugin_type, plugins in result.items():
        logger.info(f"发现 {len(plugins)} 个类型为 {plugin_type} 的插件")

    return result


def load_plugin(plugin_class: Type[PluginBase]) -> bool:
    """
    加载单个插件类

    Args:
        plugin_class: 插件类

    Returns:
        bool: 是否成功加载
    """
    try:
        return register_plugin(plugin_class)
    except Exception as e:
        logger.error(f"加载插件类时出错: {plugin_class.__name__}, 错误: {e}")
        return False


def discover_and_load_plugins(plugin_types: Optional[List[str]] = None) -> Dict[
    str, int]:
    """
    发现并加载插件

    Args:
        plugin_types: 要加载的插件类型列表，None表示加载所有类型

    Returns:
        Dict[str, int]: 每种类型加载的插件数量
    """
    result = {}

    # 发现插件
    discovered_plugins = discover_plugins()

    # 按类型加载插件
    for plugin_type, plugins in discovered_plugins.items():
        # 如果指定了类型且当前类型不在列表中，跳过
        if plugin_types is not None and plugin_type not in plugin_types:
            continue

        # 加载该类型的所有插件
        loaded_count = 0
        for plugin_class in plugins:
            if load_plugin(plugin_class):
                loaded_count += 1

        # 记录加载数量
        result[plugin_type] = loaded_count

    return result


def _find_plugin_classes(module) -> List[Type[PluginBase]]:
    """
    在模块中查找插件类

    Args:
        module: 模块对象

    Returns:
        List[Type[PluginBase]]: 插件类列表
    """
    result = []

    # 获取模块中的所有类
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # 检查是否是插件基类的子类（但不是基类本身）
        if (issubclass(obj, PluginBase)
                and obj is not PluginBase
                and obj.__module__ == module.__name__):
            result.append(obj)

    return result


def reload_plugin(plugin_id: str) -> bool:
    """
    重新加载指定的插件

    Args:
        plugin_id: 插件ID

    Returns:
        bool: 是否成功重新加载
    """
    from .plugin_registry import get_plugin_info, get_plugin_by_id

    # 获取插件信息
    plugin_info = get_plugin_info(plugin_id)
    if not plugin_info:
        logger.error(f"未找到插件: {plugin_id}")
        return False

    # 获取当前插件实例
    current_plugin = get_plugin_by_id(plugin_id)
    if current_plugin:
        # 关闭当前实例
        try:
            current_plugin.shutdown()
        except Exception as e:
            logger.error(f"关闭插件实例时出错: {plugin_id}, 错误: {e}")

    # 重新加载插件模块
    try:
        # 获取插件模块
        module_name = current_plugin.__class__.__module__
        module = sys.modules[module_name]

        # 重新加载模块
        importlib.reload(module)

        # 重新查找插件类
        plugin_classes = _find_plugin_classes(module)

        # 查找匹配的插件类
        for plugin_class in plugin_classes:
            temp_instance = plugin_class()
            if temp_instance.get_info().id == plugin_id:
                # 重新注册插件
                return register_plugin(plugin_class)

        logger.error(f"重新加载后未找到插件: {plugin_id}")
        return False
    except Exception as e:
        logger.error(f"重新加载插件时出错: {plugin_id}, 错误: {e}")
        return False
