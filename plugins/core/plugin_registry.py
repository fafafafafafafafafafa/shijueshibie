# -*- coding: utf-8 -*-
"""
插件注册表 - 管理插件的注册和查询
"""

import logging
from typing import Dict, List, Any, Optional, Type, Set

from .plugin_interface import PluginBase, PluginInfo, PluginType
from .plugin_system import get_plugin_system

logger = logging.getLogger("PluginRegistry")

# 插件注册表
_registry_initialized = False
_plugin_system = None


def init_registry() -> bool:
    """
    初始化插件注册表

    Returns:
        bool: 是否成功初始化
    """
    global _registry_initialized, _plugin_system

    if _registry_initialized:
        logger.debug("插件注册表已经初始化")
        return True

    try:
        _plugin_system = get_plugin_system()
        if not _plugin_system.initialize():
            logger.error("插件系统初始化失败")
            return False

        _registry_initialized = True
        logger.info("插件注册表初始化完成")
        return True
    except Exception as e:
        logger.error(f"初始化插件注册表失败: {e}")
        return False


def register_plugin(plugin_class: Type[PluginBase]) -> bool:
    """
    注册插件类

    Args:
        plugin_class: 插件类

    Returns:
        bool: 是否成功注册
    """
    global _plugin_system

    if not _registry_initialized:
        if not init_registry():
            return False

    return _plugin_system.register_plugin_class(plugin_class)


def create_plugin(plugin_id: str, config: Optional[Dict[str, Any]] = None) -> \
Optional[PluginBase]:
    """
    创建插件实例

    Args:
        plugin_id: 插件ID
        config: 插件配置

    Returns:
        Optional[PluginBase]: 插件实例，如果创建失败则返回None
    """
    global _plugin_system

    if not _registry_initialized:
        if not init_registry():
            return None

    return _plugin_system.create_plugin(plugin_id, config)


def create_plugin_by_type_name(plugin_type: str, plugin_name: str,
                               config: Optional[Dict[str, Any]] = None) -> \
Optional[PluginBase]:
    """
    根据类型和名称创建插件实例

    Args:
        plugin_type: 插件类型名称
        plugin_name: 插件名称
        config: 插件配置

    Returns:
        Optional[PluginBase]: 插件实例，如果创建失败则返回None
    """
    global _plugin_system

    if not _registry_initialized:
        if not init_registry():
            return None

    # 获取所有插件信息
    plugin_infos = _plugin_system.get_all_plugin_info()

    # 查找匹配的插件
    for plugin_id, info in plugin_infos.items():
        if info.type.value == plugin_type and info.name == plugin_name:
            return create_plugin(plugin_id, config)

    logger.warning(f"未找到类型为 {plugin_type} 名称为 {plugin_name} 的插件")
    return None


def get_plugin_by_id(plugin_id: str) -> Optional[PluginBase]:
    """
    按ID获取插件实例

    Args:
        plugin_id: 插件ID

    Returns:
        Optional[PluginBase]: 插件实例，如果不存在则返回None
    """
    global _plugin_system

    if not _registry_initialized:
        if not init_registry():
            return None

    return _plugin_system.get_plugin(plugin_id)


def get_plugin_info(plugin_id: str) -> Optional[PluginInfo]:
    """
    获取插件信息

    Args:
        plugin_id: 插件ID

    Returns:
        Optional[PluginInfo]: 插件信息，如果不存在则返回None
    """
    global _plugin_system

    if not _registry_initialized:
        if not init_registry():
            return None

    return _plugin_system.get_plugin_info(plugin_id)


def get_plugins(plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    获取所有插件或指定类型的插件

    Args:
        plugin_type: 插件类型，None表示获取所有类型

    Returns:
        List[Dict[str, Any]]: 插件信息列表
    """
    global _plugin_system

    if not _registry_initialized:
        if not init_registry():
            return []

    result = []
    plugin_infos = _plugin_system.get_all_plugin_info()

    for plugin_id, info in plugin_infos.items():
        if plugin_type is None or info.type.value == plugin_type:
            result.append(info.to_dict())

    return result


def get_plugin_by_capability(capability: str) -> List[PluginBase]:
    """
    获取具有指定能力的所有插件

    Args:
        capability: 能力名称

    Returns:
        List[PluginBase]: 插件实例列表
    """
    global _plugin_system

    if not _registry_initialized:
        if not init_registry():
            return []

    result = []
    plugin_infos = _plugin_system.get_all_plugin_info()

    for plugin_id, info in plugin_infos.items():
        if capability in info.capabilities:
            plugin = get_plugin_by_id(plugin_id)
            if plugin:
                result.append(plugin)

    return result


def shutdown_registry() -> bool:
    """
    关闭插件注册表，释放资源

    Returns:
        bool: 是否成功关闭
    """
    global _registry_initialized, _plugin_system

    if not _registry_initialized:
        return True

    success = _plugin_system.shutdown()
    _registry_initialized = False

    logger.info("插件注册表已关闭")
    return success
