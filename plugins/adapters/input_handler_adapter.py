# -*- coding: utf-8 -*-
"""
输入处理器适配器模块 - 将现有的 InputHandler 类适配为插件系统中的插件

此适配器允许输入处理器作为标准插件集成到插件系统中，遵循插件接口规范，
同时保留其所有原始功能。
"""

import logging
from typing import Dict, Any, Optional, Tuple, Callable

# 导入基础插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    InputHandlerPluginInterface
)

# 日志配置
from utils.logger_config import get_logger

logger = get_logger("InputHandlerAdapter")


class InputHandlerAdapter(InputHandlerPluginInterface):
    """
    输入处理器适配器类，将InputHandler适配为符合InputHandlerPluginInterface的插件

    此适配器遵循适配器设计模式，封装现有输入处理器组件并提供统一的插件接口。
    """

    def __init__(self, input_handler=None, plugin_id="input_handler_plugin",
                 plugin_config=None):
        """
        初始化输入处理器适配器

        Args:
            input_handler: 现有输入处理器实例，如果为None则会在initialize时创建
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "Input Handler Plugin"
        self._version = "1.0.0"
        self._description = "将输入处理器适配为标准插件"
        self._config = plugin_config or {}

        # 适配的输入处理器实例
        self._input_handler = input_handler

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 事件系统引用
        self._event_system = None

        # 处理器映射
        self._handler_mapping = {}  # {plugin_key: original_key}

        logger.info(f"输入处理器适配器已创建: {plugin_id}")

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
        """获取插件依赖项列表"""
        return ['display_manager_plugin']  # 输入处理器通常依赖于显示管理器

    def is_initialized(self) -> bool:
        """检查插件是否已初始化"""
        return self._initialized

    def is_enabled(self) -> bool:
        """检查插件是否已启用"""
        return self._enabled

    def initialize(self, context: Dict[str, Any] = None) -> bool:
        """初始化插件

        Args:
            context: 插件初始化上下文，可以包含共享资源

        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.is_initialized():
                logger.warning(f"插件 {self._id} 已初始化，跳过")
                return True

            display_manager = None

            # 从上下文中获取依赖
            if context:
                # 获取事件系统
                if 'event_system' in context:
                    self._event_system = context['event_system']

                # 获取显示管理器
                if 'display_manager' in context:
                    display_manager = context['display_manager']
                elif 'display_manager_plugin' in context:
                    display_manager_plugin = context['display_manager_plugin']
                    if hasattr(display_manager_plugin, 'get_display_manager'):
                        display_manager = display_manager_plugin.get_display_manager()

            # 如果没有输入处理器实例，尝试创建一个
            if self._input_handler is None:
                from input_handler import InputHandler

                if display_manager is None:
                    logger.warning("未找到显示管理器，创建输入处理器可能会受限")

                self._input_handler = InputHandler(
                    display_manager=display_manager
                )
                logger.info(f"已创建新的输入处理器实例")
            else:
                # 如果已有实例但没有显示管理器，尝试设置
                if display_manager and hasattr(self._input_handler,
                                               'set_display_manager'):
                    self._input_handler.set_display_manager(display_manager)
                    logger.info(f"已设置输入处理器的显示管理器")

            # 设置事件系统
            if self._event_system and hasattr(self._input_handler,
                                              'set_event_system'):
                self._input_handler.set_event_system(self._event_system)
                logger.info(f"已设置输入处理器的事件系统")

            self._initialized = True
            logger.info(f"插件 {self._id} 初始化成功")
            return True

        except Exception as e:
            logger.error(f"插件 {self._id} 初始化失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def enable(self) -> bool:
        """启用插件

        Returns:
            bool: 启用是否成功
        """
        if not self.is_initialized():
            logger.error(f"插件 {self._id} 未初始化，无法启用")
            return False

        try:
            # 执行启用逻辑
            self._enabled = True
            logger.info(f"插件 {self._id} 已启用")
            return True
        except Exception as e:
            logger.error(f"启用插件 {self._id} 时出错: {e}")
            return False

    def disable(self) -> bool:
        """禁用插件

        Returns:
            bool: 禁用是否成功
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 已经是禁用状态")
            return True

        try:
            # 执行禁用逻辑
            self._enabled = False
            logger.info(f"插件 {self._id} 已禁用")
            return True
        except Exception as e:
            logger.error(f"禁用插件 {self._id} 时出错: {e}")
            return False

    def configure(self, config: Dict[str, Any]) -> bool:
        """配置插件

        Args:
            config: 插件配置参数

        Returns:
            bool: 配置是否成功
        """
        try:
            # 更新配置
            self._config.update(config)

            # 输入处理器通常没有专门的配置方法，但可以根据需要添加特定逻辑
            logger.info(f"插件 {self._id} 配置已更新")
            return True
        except Exception as e:
            logger.error(f"配置插件 {self._id} 时出错: {e}")
            return False

    def cleanup(self) -> bool:
        """清理插件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            # 执行清理逻辑
            self._enabled = False
            self._initialized = False

            # 调用输入处理器的清理方法
            if self._input_handler and hasattr(self._input_handler, 'cleanup'):
                self._input_handler.cleanup()

            # 清除引用
            self._event_system = None

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 实现InputHandlerPluginInterface特定方法 =============

    def process_input(self) -> bool:
        """
        处理用户输入

        Returns:
            bool: 是否请求退出程序
        """
        if not self.is_enabled() or not self._input_handler:
            return False

        try:
            return self._input_handler.process_input()
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")
            return False

    def register_handler(self, key: str, callback: Callable,
                         description: str = None) -> bool:
        """
        注册按键处理函数

        Args:
            key: 按键字符
            callback: 回调函数
            description: 按键功能描述

        Returns:
            bool: 是否成功注册
        """
        if not self.is_enabled() or not self._input_handler:
            logger.warning(f"插件 {self._id} 未启用或未初始化，无法注册处理函数")
            return False

        try:
            # 将此插件的处理函数注册到实际的输入处理器
            success = self._input_handler.register_handler(key, callback,
                                                           description)

            # 记录映射关系
            if success:
                self._handler_mapping[f"{self._id}:{key}"] = key

            return success
        except Exception as e:
            logger.error(f"注册按键处理函数时出错: {e}")
            return False

    def register_feature_toggle(self, feature_key: str,
                                callback: Callable) -> bool:
        """
        注册功能切换回调

        Args:
            feature_key: 功能对应的按键
            callback: 切换回调函数

        Returns:
            bool: 是否成功注册
        """
        if not self.is_enabled() or not self._input_handler:
            logger.warning(f"插件 {self._id} 未启用或未初始化，无法注册功能切换")
            return False

        try:
            # 将此插件的功能切换回调注册到实际的输入处理器
            if hasattr(self._input_handler, 'register_feature_toggle'):
                success = self._input_handler.register_feature_toggle(
                    feature_key, callback)
                return success
            else:
                logger.error("输入处理器没有实现register_feature_toggle方法")
                return False
        except Exception as e:
            logger.error(f"注册功能切换回调时出错: {e}")
            return False

    def is_exit_requested(self) -> bool:
        """
        检查是否请求退出

        Returns:
            bool: 是否请求退出
        """
        if not self._input_handler:
            return False

        try:
            if hasattr(self._input_handler, 'is_exit_requested'):
                return self._input_handler.is_exit_requested()
            return False
        except Exception as e:
            logger.error(f"检查退出请求时出错: {e}")
            return False

    def show_help(self) -> None:
        """显示帮助信息"""
        if not self.is_enabled() or not self._input_handler:
            logger.warning(f"插件 {self._id} 未启用或未初始化，无法显示帮助")
            return

        try:
            if hasattr(self._input_handler, 'show_help'):
                self._input_handler.show_help()
        except Exception as e:
            logger.error(f"显示帮助时出错: {e}")

    def get_input_handler(self):
        """
        获取原始输入处理器实例

        Returns:
            原始输入处理器实例
        """
        return self._input_handler

    def set_input_handler(self, input_handler):
        """
        设置输入处理器实例

        Args:
            input_handler: 新的输入处理器实例
        """
        self._input_handler = input_handler

        # 如果已设置事件系统，同时设置给新输入处理器
        if self._event_system and hasattr(input_handler, 'set_event_system'):
            input_handler.set_event_system(self._event_system)

        logger.info(f"已更新输入处理器实例")

    def get_help_text(self) -> Dict[str, str]:
        """
        获取帮助文本

        Returns:
            Dict: 按键帮助文本 {key: description}
        """
        if not self._input_handler:
            return {}

        try:
            if hasattr(self._input_handler, 'help_text'):
                return self._input_handler.help_text
            return {}
        except Exception as e:
            logger.error(f"获取帮助文本时出错: {e}")
            return {}


# 插件系统工厂方法
def create_plugin(plugin_id="input_handler_plugin", config=None,
                  context=None) -> PluginInterface:
    """
    创建输入处理器插件实例

    此函数是插件系统识别和加载插件的入口点

    Args:
        plugin_id: 插件唯一标识符
        config: 插件配置
        context: 插件上下文

    Returns:
        PluginInterface: 插件实例
    """
    try:
        # 创建适配器实例
        plugin = InputHandlerAdapter(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建输入处理器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建输入处理器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
