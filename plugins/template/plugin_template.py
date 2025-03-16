# -*- coding: utf-8 -*-
"""
插件模板模块 - 提供插件开发的基础模板

此模块为插件开发者提供了标准结构和接口实现的参考示例，
可以作为创建新插件的起点。
"""

import logging
import time
from typing import Dict, Any, List, Optional

# 导入插件接口
from plugins.core.plugin_interface import PluginInterface

# 导入日志配置
from utils.logger_config import get_logger

logger = get_logger("TemplatePlugin")


class TemplatePlugin(PluginInterface):
    """
    插件模板类 - 提供插件的基本结构和方法实现

    此类实现了PluginInterface接口，为开发新插件提供了
    标准参考模板，包含完整的生命周期方法和事件处理。

    开发新插件时，可以基于此模板进行修改和扩展。
    """

    def __init__(self, plugin_id="template_plugin", plugin_config=None):
        """
        初始化插件模板

        Args:
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "Template Plugin"
        self._version = "1.0.0"
        self._description = "插件开发模板，提供基础结构和接口实现"
        self._config = plugin_config or {}

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 插件特定属性
        self._sample_property = self._config.get('sample_property',
                                                 'default_value')
        self._execution_count = 0
        self._last_execution_time = 0

        # 事件系统引用
        self._event_system = None

        logger.info(f"模板插件已创建: {plugin_id}")

    # ============= 实现PluginInterface基本方法 =============

    @property
    def id(self) -> str:
        """获取插件ID"""
        return self._id

    @property
    def name(self) -> str:
        """获取插件名称"""
        return self._name

    @property
    def version(self) -> str:
        """获取插件版本"""
        return self._version

    @property
    def description(self) -> str:
        """获取插件描述"""
        return self._description

    def get_dependencies(self) -> list:
        """
        获取插件依赖项列表

        返回此插件依赖的其他插件ID列表。如果没有依赖，返回空列表。

        Returns:
            list: 依赖的插件ID列表
        """
        # 返回依赖的插件ID列表，如果没有依赖则返回空列表
        return []

    def is_initialized(self) -> bool:
        """
        检查插件是否已初始化

        Returns:
            bool: 插件是否已初始化
        """
        return self._initialized

    def is_enabled(self) -> bool:
        """
        检查插件是否已启用

        Returns:
            bool: 插件是否已启用
        """
        return self._enabled

    def initialize(self, context: Dict[str, Any] = None) -> bool:
        """
        初始化插件

        在这里执行插件的初始化逻辑，如加载资源、设置事件订阅等。

        Args:
            context: 插件初始化上下文，可以包含共享资源

        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.is_initialized():
                logger.warning(f"插件 {self._id} 已初始化，跳过")
                return True

            # 在这里执行初始化逻辑
            logger.info(f"执行插件 {self._id} 的初始化逻辑")

            # 示例：从上下文中获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']
                logger.info("已设置事件系统")

                # 设置事件订阅
                if hasattr(self._event_system, 'subscribe'):
                    self._setup_event_subscriptions()

                # 发布初始化事件
                if hasattr(self._event_system, 'publish'):
                    self._event_system.publish(
                        "plugin_initialized",
                        {
                            'plugin_id': self._id,
                            'plugin_type': 'template'
                        }
                    )

            # 初始化成功后设置标志
            self._initialized = True
            logger.info(f"插件 {self._id} 初始化成功")
            return True

        except Exception as e:
            logger.error(f"插件 {self._id} 初始化失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def enable(self) -> bool:
        """
        启用插件

        在这里执行插件启用时的逻辑，激活插件功能。

        Returns:
            bool: 启用是否成功
        """
        if not self.is_initialized():
            logger.error(f"插件 {self._id} 未初始化，无法启用")
            return False

        try:
            # 执行启用逻辑
            logger.info(f"执行插件 {self._id} 的启用逻辑")

            # 更新状态
            self._enabled = True

            # 发布启用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_enabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'template'
                    }
                )

            logger.info(f"插件 {self._id} 已启用")
            return True
        except Exception as e:
            logger.error(f"启用插件 {self._id} 时出错: {e}")
            return False

    def disable(self) -> bool:
        """
        禁用插件

        在这里执行插件禁用时的逻辑，停止插件功能，但保留状态。

        Returns:
            bool: 禁用是否成功
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 已经是禁用状态")
            return True

        try:
            # 执行禁用逻辑
            logger.info(f"执行插件 {self._id} 的禁用逻辑")

            # 更新状态
            self._enabled = False

            # 发布禁用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_disabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'template'
                    }
                )

            logger.info(f"插件 {self._id} 已禁用")
            return True
        except Exception as e:
            logger.error(f"禁用插件 {self._id} 时出错: {e}")
            return False

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        配置插件

        更新插件配置参数，可以在运行时动态修改插件行为。

        Args:
            config: 插件配置参数

        Returns:
            bool: 配置是否成功
        """
        try:
            # 更新配置
            self._config.update(config)
            logger.info(f"已更新插件 {self._id} 配置")

            # 根据新配置更新插件属性
            if 'sample_property' in config:
                old_value = self._sample_property
                self._sample_property = config['sample_property']
                logger.info(
                    f"属性已更新: {old_value} -> {self._sample_property}")

            # 可以添加其他配置处理逻辑

            return True
        except Exception as e:
            logger.error(f"配置插件 {self._id} 时出错: {e}")
            return False

    def cleanup(self) -> bool:
        """
        清理插件资源

        在插件卸载前执行清理工作，释放资源，保存状态等。

        Returns:
            bool: 清理是否成功
        """
        try:
            # 执行清理逻辑
            logger.info(f"执行插件 {self._id} 的清理逻辑")

            # 取消事件订阅
            if self._event_system and hasattr(self._event_system,
                                              'unsubscribe_all'):
                self._event_system.unsubscribe_all(self._id)

            # 重置状态
            self._enabled = False
            self._initialized = False

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 插件特定方法 =============

    def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行插件特定任务

        这是一个示例方法，展示插件如何定义和实现自己的功能。

        Args:
            task_data: 任务数据

        Returns:
            Dict: 任务结果
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法执行任务")
            return {'error': 'Plugin not enabled'}

        try:
            # 记录执行信息
            self._execution_count += 1
            self._last_execution_time = time.time()

            # 执行任务逻辑
            logger.info(f"执行任务: {task_data}")

            # 示例任务处理
            result = {
                'status': 'success',
                'execution_count': self._execution_count,
                'timestamp': self._last_execution_time,
                'input_processed': task_data
            }

            # 添加示例处理结果
            if 'value' in task_data:
                result['processed_value'] = task_data['value'] * 2

            # 发布任务完成事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "task_completed",
                    {
                        'plugin_id': self._id,
                        'task_id': task_data.get('id', 'unknown'),
                        'result': result
                    }
                )

            return result

        except Exception as e:
            logger.error(f"执行任务时出错: {e}")
            return {'error': str(e)}

    def get_plugin_status(self) -> Dict[str, Any]:
        """
        获取插件状态信息

        返回插件当前状态的详细信息，可用于监控和调试。

        Returns:
            Dict: 插件状态信息
        """
        return {
            'id': self._id,
            'name': self._name,
            'version': self._version,
            'initialized': self._initialized,
            'enabled': self._enabled,
            'execution_count': self._execution_count,
            'last_execution_time': self._last_execution_time,
            'sample_property': self._sample_property,
            'config': self._config
        }

    # ============= 辅助方法 =============

    def _setup_event_subscriptions(self):
        """设置事件订阅"""
        if not self._event_system:
            return

        try:
            # 示例：订阅相关事件
            self._event_system.subscribe("system_status_changed",
                                         self._on_system_status_changed)
            self._event_system.subscribe("plugin_message",
                                         self._on_plugin_message)

            logger.info("已设置事件订阅")
        except Exception as e:
            logger.error(f"设置事件订阅时出错: {e}")

    # ============= 事件处理方法 =============

    def _on_system_status_changed(self, data):
        """
        处理系统状态变更事件

        Args:
            data: 事件数据
        """
        logger.info(f"收到系统状态变更事件: {data}")
        # 处理系统状态变更逻辑

    def _on_plugin_message(self, data):
        """
        处理插件消息事件

        Args:
            data: 事件数据
        """
        # 只处理发给此插件的消息
        if data.get('target_plugin_id') == self._id:
            logger.info(f"收到插件消息: {data.get('message', '')}")
            # 处理消息逻辑


# 插件系统工厂方法
def create_plugin(plugin_id="template_plugin", config=None,
                  context=None) -> PluginInterface:
    """
    创建插件实例

    此函数是插件系统识别和加载插件的入口点。
    对于新的插件实现，只需修改插件类和相关参数即可。

    Args:
        plugin_id: 插件唯一标识符
        config: 插件配置
        context: 插件上下文

    Returns:
        PluginInterface: 插件实例
    """
    try:
        # 创建插件实例
        plugin = TemplatePlugin(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建模板插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建模板插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
