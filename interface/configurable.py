# -*- coding: utf-8 -*-
"""
可配置组件接口模块 - 定义配置管理和验证功能
提供组件配置的加载、验证和持久化能力
"""

import logging
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from core.component_interface import ComponentInterface, BaseComponent

logger = logging.getLogger("Configurable")

class ConfigurableComponentInterface(ABC):
    """
    可配置组件接口 - 定义配置管理和验证方法
    """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        获取组件配置

        Returns:
            Dict[str, Any]: 组件配置
        """
        pass

    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            config: 新配置

        Returns:
            bool: 是否成功更新配置
        """
        pass

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取组件配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证配置是否有效

        Args:
            config: 要验证的配置

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        pass


class BaseConfigurableComponent(BaseComponent, ConfigurableComponentInterface):
    """
    基础可配置组件 - 提供可配置组件接口的基本实现
    实现组件的配置管理功能
    """

    def __init__(self, component_id: str, component_type: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化基础可配置组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
            config: 初始配置
        """
        super().__init__(component_id, component_type)

        # 初始化配置
        self._config = config or {}

        # 配置模式
        self._config_schema = self._create_config_schema()

    def get_config(self) -> Dict[str, Any]:
        """
        获取组件配置

        Returns:
            Dict[str, Any]: 组件配置
        """
        return dict(self._config)

    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            config: 新配置

        Returns:
            bool: 是否成功更新配置
        """
        try:
            # 验证配置
            valid, error = self.validate_config(config)
            if not valid:
                logger.warning(
                    f"配置验证失败: {self._component_id}, 错误: {error}")
                return False

            # 更新配置
            old_config = dict(self._config)
            self._config.update(config)

            # 应用配置
            result = self._apply_config(old_config, self._config)
            if not result:
                # 恢复旧配置
                self._config = old_config
                logger.warning(f"应用配置失败: {self._component_id}")
                return False

            logger.debug(f"组件配置已更新: {self._component_id}")
            return True
        except Exception as e:
            logger.error(f"更新配置错误: {self._component_id}, 错误: {e}")
            return False

    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取组件配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        return dict(self._config_schema)

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证配置是否有效

        Args:
            config: 要验证的配置

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        try:
            # 默认实现：检查必需字段
            schema = self.get_config_schema()

            for field_name, field_info in schema.items():
                if field_info.get('required', False) and field_name not in config:
                    return False, f"缺少必需字段: {field_name}"

            return True, None
        except Exception as e:
            return False, str(e)

    def _create_config_schema(self) -> Dict[str, Any]:
        """
        创建配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        # 子类应重写此方法
        return {}

    def _apply_config(self, old_config: Dict[str, Any],
                      new_config: Dict[str, Any]) -> bool:
        """
        应用新配置

        Args:
            old_config: 旧配置
            new_config: 新配置

        Returns:
            bool: 是否成功应用配置
        """
        # 子类应重写此方法
        return True
