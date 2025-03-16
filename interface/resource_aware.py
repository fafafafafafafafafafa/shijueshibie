# -*- coding: utf-8 -*-
"""
资源感知组件接口模块 - 定义资源管理和适应能力
提供组件资源使用监控和自适应功能
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from core.component_interface import ComponentInterface, BaseComponent
from core.resource_manager import ResourceManager, AdaptationLevel

logger = logging.getLogger("ResourceAware")

class ResourceAwareComponentInterface(ABC):
    """
    资源感知组件接口 - 支持资源管理的组件接口
    定义组件的资源管理方法
    """

    @abstractmethod
    def get_resource_requirements(self) -> Dict[str, Any]:
        """
        获取组件资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        pass

    @abstractmethod
    def adapt_to_resources(self, available_resources: Dict[str, Any]) -> bool:
        """
        适应可用资源

        Args:
            available_resources: 可用资源

        Returns:
            bool: 是否成功适应
        """
        pass

    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取组件资源使用情况

        Returns:
            Dict[str, Any]: 资源使用情况
        """
        pass


class BaseResourceAwareComponent(BaseComponent, ResourceAwareComponentInterface):
    """
    基础资源感知组件 - 提供资源感知组件接口的基本实现
    实现组件的资源管理功能
    """

    def __init__(self, component_id: str, component_type: Optional[str] = None):
        """
        初始化基础资源感知组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
        """
        super().__init__(component_id, component_type)

        # 获取资源管理器
        from core.resource_manager import ResourceManager
        self._resource_manager = ResourceManager.get_instance()

        # 资源分配ID列表
        self._resource_allocations = []

        # 资源需求
        self._resource_requirements = self._create_resource_requirements()

    def get_resource_requirements(self) -> Dict[str, Any]:
        """
        获取组件资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        return dict(self._resource_requirements)

    def adapt_to_resources(self, available_resources: Dict[str, Any]) -> bool:
        """
        适应可用资源

        Args:
            available_resources: 可用资源

        Returns:
            bool: 是否成功适应
        """
        try:
            # 获取适应建议
            adaptation_level = self._resource_manager.get_adaptation_level()
            suggestions = self._resource_manager.get_adaptation_suggestions()

            # 应用适应策略
            return self._apply_resource_adaptation(adaptation_level, suggestions)
        except Exception as e:
            logger.error(f"资源适应错误: {self._component_id}, 错误: {e}")
            return False

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取组件资源使用情况

        Returns:
            Dict[str, Any]: 资源使用情况
        """
        # A default implementation that returns empty dictionary
        return {}

    def _create_resource_requirements(self) -> Dict[str, Any]:
        """
        创建资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        # 子类应重写此方法
        return {}

    def _apply_resource_adaptation(self, adaptation_level: AdaptationLevel, suggestions: Dict[str, Any]) -> bool:
        """
        应用资源适应策略

        Args:
            adaptation_level: 适应级别
            suggestions: 适应建议

        Returns:
            bool: 是否成功适应
        """
        # 子类应重写此方法
        return True
