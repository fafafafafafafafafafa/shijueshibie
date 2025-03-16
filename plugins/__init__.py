# -*- coding: utf-8 -*-
"""
插件系统模块 - 提供动态插件加载、注册和管理功能
支持多种插件类型，包括检测器、识别器、映射器和可视化器
"""

import importlib
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Type, Set

# 设置日志记录器
logger = logging.getLogger("PluginSystem")

# 导出版本信息
__version__ = "1.0.0"

# 插件系统状态
_initialized = False
_plugins_loaded = False


def init_plugin_system():
    """
    初始化插件系统，必须在使用任何插件功能前调用

    Returns:
        bool: 初始化是否成功
    """
    global _initialized

    if _initialized:
        logger.debug("插件系统已经初始化")
        return True

    try:
        # 导入核心模块
        from .core import plugin_system, plugin_registry

        # 初始化插件注册表
        plugin_registry.init_registry()

        # 设置插件搜索路径
        _setup_plugin_paths()

        _initialized = True
        logger.info("插件系统初始化完成")
        return True
    except Exception as e:
        logger.error(f"插件系统初始化失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def _setup_plugin_paths():
    """设置插件搜索路径"""
    # 获取当前模块的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 添加插件目录到Python路径
    plugin_dirs = [
        os.path.join(current_dir, "detectors"),
        os.path.join(current_dir, "recognizers"),
        os.path.join(current_dir, "mappers"),
        os.path.join(current_dir, "visualizers"),
        os.path.join(current_dir, "tools"),
    ]

    # 添加外部插件目录（如果存在）
    user_plugin_dir = os.path.expanduser("~/.pose_tracking/plugins")
    if os.path.exists(user_plugin_dir):
        plugin_dirs.append(user_plugin_dir)

    # 将这些路径添加到sys.path
    for plugin_dir in plugin_dirs:
        if os.path.exists(plugin_dir) and plugin_dir not in sys.path:
            sys.path.append(plugin_dir)
            logger.debug(f"添加插件目录到路径: {plugin_dir}")


def load_plugins(plugin_types: Optional[List[str]] = None) -> Dict[str, int]:
    """
    加载所有可用插件或指定类型的插件

    Args:
        plugin_types: 要加载的插件类型列表，None表示加载所有类型

    Returns:
        Dict[str, int]: 每种类型加载的插件数量
    """
    global _plugins_loaded

    if not _initialized:
        if not init_plugin_system():
            return {}

    try:
        from .core import plugin_loader, plugin_registry

        # 加载适配器插件（将现有组件适配为插件）
        adapter_count = _load_adapters()

        # 加载标准插件
        plugin_counts = plugin_loader.discover_and_load_plugins(plugin_types)
        plugin_counts['adapters'] = adapter_count

        _plugins_loaded = True

        # 记录加载结果
        total_plugins = sum(plugin_counts.values())
        logger.info(f"已加载 {total_plugins} 个插件: {plugin_counts}")

        return plugin_counts
    except Exception as e:
        logger.error(f"加载插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}


def _load_adapters() -> int:
    """
    加载适配器插件

    Returns:
        int: 加载的适配器数量
    """
    from .core import plugin_registry

    # 获取适配器目录
    adapters_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "adapters")

    if not os.path.exists(adapters_dir):
        logger.warning(f"适配器目录不存在: {adapters_dir}")
        return 0

    # 导入适配器模块
    count = 0
    for filename in os.listdir(adapters_dir):
        if filename.endswith("_adapter.py"):
            module_name = filename[:-3]  # 去除.py扩展名
            try:
                # 动态导入适配器模块
                module = importlib.import_module(f".adapters.{module_name}",
                                                 package="plugins")

                # 尝试调用模块的register_adapters函数
                if hasattr(module, "register_adapters"):
                    registered = module.register_adapters()
                    count += registered
                    logger.debug(
                        f"已注册 {registered} 个适配器，来自 {module_name}")
            except Exception as e:
                logger.error(f"加载适配器 {module_name} 失败: {e}")

    logger.info(f"已加载 {count} 个适配器插件")
    return count


def get_available_plugins(plugin_type: Optional[str] = None) -> List[
    Dict[str, Any]]:
    """
    获取所有可用插件或指定类型的插件

    Args:
        plugin_type: 插件类型，None表示获取所有类型

    Returns:
        List[Dict[str, Any]]: 插件信息列表
    """
    if not _plugins_loaded:
        load_plugins()

    from .core import plugin_registry
    return plugin_registry.get_plugins(plugin_type)


def get_plugin_by_id(plugin_id: str) -> Optional[Any]:
    """
    按ID获取插件实例

    Args:
        plugin_id: 插件ID

    Returns:
        Optional[Any]: 插件实例，如果不存在则返回None
    """
    if not _plugins_loaded:
        load_plugins()

    from .core import plugin_registry
    return plugin_registry.get_plugin_by_id(plugin_id)


def create_plugin_instance(plugin_type: str, plugin_name: str,
                           config: Optional[Dict[str, Any]] = None) -> Optional[
    Any]:
    """
    创建指定类型和名称的插件实例

    Args:
        plugin_type: 插件类型
        plugin_name: 插件名称
        config: 可选的插件配置

    Returns:
        Optional[Any]: 插件实例，如果创建失败则返回None
    """
    if not _plugins_loaded:
        load_plugins()

    from .core import plugin_registry
    return plugin_registry.create_plugin(plugin_type, plugin_name, config)


# 导出主要功能
__all__ = [
    'init_plugin_system',
    'load_plugins',
    'get_available_plugins',
    'get_plugin_by_id',
    'create_plugin_instance',
]
