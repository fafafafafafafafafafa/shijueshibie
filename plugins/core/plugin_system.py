# -*- coding: utf-8 -*-
"""
插件系统实现 - 负责插件系统的核心功能
"""

import logging
import threading
from typing import Dict, List, Any, Optional, Type

from .plugin_interface import PluginBase, PluginInfo, PluginType

logger = logging.getLogger("PluginSystem")


class PluginSystem:
    """
    插件系统类 - 管理插件的核心类
    """

    _instance = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> 'PluginSystem':
        """获取PluginSystem单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = PluginSystem()
            return cls._instance

    def __init__(self):
        """
        初始化插件系统
        """
        if PluginSystem._instance is not None:
            logger.warning("PluginSystem是单例类，请使用get_instance()获取实例")
            return

        # 初始化状态
        self._initialized = False

        # 插件实例字典 {plugin_id: plugin_instance}
        self._plugin_instances = {}

        # 插件类字典 {plugin_id: plugin_class}
        self._plugin_classes = {}

        # 插件类型映射 {plugin_type: {plugin_id}}
        self._plugin_types = {}

        # 插件信息映射 {plugin_id: PluginInfo}
        self._plugin_info = {}

        # 插件依赖图 {plugin_id: [dependency_id]}
        self._dependency_graph = {}

        # 插件配置 {plugin_id: config}
        self._plugin_configs = {}

        logger.info("插件系统已创建")

    def initialize(self) -> bool:
        """
        初始化插件系统

        Returns:
            bool: 是否成功初始化
        """
        if self._initialized:
            logger.debug("插件系统已经初始化")
            return True

        try:
            # 这里可以添加初始化逻辑

            self._initialized = True
            logger.info("插件系统初始化完成")
            return True
        except Exception as e:
            logger.error(f"插件系统初始化失败: {e}")
            return False

    def register_plugin_class(self, plugin_class: Type[PluginBase]) -> bool:
        """
        注册插件类

        Args:
            plugin_class: 插件类

        Returns:
            bool: 是否成功注册
        """
        try:
            # 创建临时实例以获取插件信息
            temp_instance = plugin_class()
            plugin_info = temp_instance.get_info()

            plugin_id = plugin_info.id
            plugin_type = plugin_info.type

            # 检查是否已注册
            if plugin_id in self._plugin_classes:
                logger.warning(f"插件ID '{plugin_id}' 已注册，将被覆盖")

            # 添加到插件类字典
            self._plugin_classes[plugin_id] = plugin_class

            # 添加到插件信息字典
            self._plugin_info[plugin_id] = plugin_info

            # 添加到类型映射
            if plugin_type not in self._plugin_types:
                self._plugin_types[plugin_type] = set()
            self._plugin_types[plugin_type].add(plugin_id)

            # 添加到依赖图
            self._dependency_graph[plugin_id] = plugin_info.dependencies

            logger.info(f"已注册插件类: {plugin_id}, 类型: {plugin_type.value}")
            return True
        except Exception as e:
            logger.error(f"注册插件类失败: {e}")
            return False

    def create_plugin(self, plugin_id: str,
                      config: Optional[Dict[str, Any]] = None) -> Optional[
        PluginBase]:
        """
        创建插件实例

        Args:
            plugin_id: 插件ID
            config: 插件配置

        Returns:
            Optional[PluginBase]: 插件实例，如果创建失败则返回None
        """
        if plugin_id not in self._plugin_classes:
            logger.error(f"未找到插件ID: {plugin_id}")
            return None

        try:
            # 创建插件实例
            plugin_class = self._plugin_classes[plugin_id]
            plugin = plugin_class()

            # 检查依赖是否满足
            if not self._check_dependencies(plugin_id):
                logger.error(f"插件 {plugin_id} 的依赖未满足")
                return None

            # 初始化插件
            if not plugin.initialize(config):
                logger.error(f"插件 {plugin_id} 初始化失败")
                return None

            # 保存插件实例和配置
            self._plugin_instances[plugin_id] = plugin
            if config:
                self._plugin_configs[plugin_id] = config

            logger.info(f"已创建插件实例: {plugin_id}")
            return plugin
        except Exception as e:
            logger.error(f"创建插件实例失败: {plugin_id}, 错误: {e}")
            return None

    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """
        获取插件实例

        Args:
            plugin_id: 插件ID

        Returns:
            Optional[PluginBase]: 插件实例，如果不存在则返回None
        """
        # 如果插件实例已存在，直接返回
        if plugin_id in self._plugin_instances:
            return self._plugin_instances[plugin_id]

        # 如果插件类存在但实例不存在，创建实例
        if plugin_id in self._plugin_classes:
            config = self._plugin_configs.get(plugin_id)
            return self.create_plugin(plugin_id, config)

        logger.warning(f"未找到插件: {plugin_id}")
        return None

    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """
        获取插件信息

        Args:
            plugin_id: 插件ID

        Returns:
            Optional[PluginInfo]: 插件信息，如果不存在则返回None
        """
        return self._plugin_info.get(plugin_id)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """
        获取指定类型的所有插件ID

        Args:
            plugin_type: 插件类型

        Returns:
            List[str]: 插件ID列表
        """
        if plugin_type not in self._plugin_types:
            return []
        return list(self._plugin_types[plugin_type])

    def get_all_plugin_info(self) -> Dict[str, PluginInfo]:
        """
        获取所有插件信息

        Returns:
            Dict[str, PluginInfo]: 插件信息字典
        """
        return dict(self._plugin_info)

    def _check_dependencies(self, plugin_id: str) -> bool:
        """
        检查插件依赖是否满足

        Args:
            plugin_id: 插件ID

        Returns:
            bool: 依赖是否满足
        """
        if plugin_id not in self._dependency_graph:
            return True  # 没有依赖关系

        dependencies = self._dependency_graph[plugin_id]
        for dep_id in dependencies:
            if dep_id not in self._plugin_classes:
                logger.error(f"插件 {plugin_id} 依赖的插件 {dep_id} 不存在")
                return False

            # 如果依赖插件未实例化，尝试实例化
            if dep_id not in self._plugin_instances:
                dep_config = self._plugin_configs.get(dep_id)
                if not self.create_plugin(dep_id, dep_config):
                    logger.error(f"无法创建依赖插件: {dep_id}")
                    return False

        return True  # 所有依赖已满足

    def shutdown(self) -> bool:
        """
        关闭插件系统，销毁所有插件实例

        Returns:
            bool: 是否成功关闭
        """
        if not self._initialized:
            logger.warning("插件系统尚未初始化")
            return True

        success = True

        # 按依赖关系的反序关闭插件
        for plugin_id, plugin in list(self._plugin_instances.items()):
            try:
                if not plugin.shutdown():
                    logger.warning(f"插件 {plugin_id} 关闭失败")
                    success = False
                else:
                    logger.debug(f"插件 {plugin_id} 已关闭")
            except Exception as e:
                logger.error(f"关闭插件 {plugin_id} 时出错: {e}")
                success = False

        # 清除状态
        self._plugin_instances.clear()
        self._initialized = False

        logger.info("插件系统已关闭")
        return success


# 便捷函数 - 获取插件系统单例
def get_plugin_system() -> PluginSystem:
    """获取插件系统单例实例"""
    return PluginSystem.get_instance()
