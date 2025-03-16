# -*- coding: utf-8 -*-
"""
可视化器适配器模块 - 将现有的 EnhancedVisualizer 类适配为插件系统中的插件

此适配器允许可视化器作为标准插件集成到插件系统中，遵循插件接口规范，
同时保留其所有原始功能。
"""

import logging
from typing import Dict, Any, Optional, Tuple

# 导入基础插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    VisualizerPluginInterface
)

# 日志配置
from utils.logger_config import get_logger

logger = get_logger("VisualizerAdapter")


class VisualizerAdapter(VisualizerPluginInterface):
    """
    可视化器适配器类，将EnhancedVisualizer适配为符合VisualizerPluginInterface的插件

    此适配器遵循适配器设计模式，封装现有可视化器组件并提供统一的插件接口。
    """

    def __init__(self, visualizer=None, plugin_id="visualizer_plugin",
                 plugin_config=None):
        """
        初始化可视化器适配器

        Args:
            visualizer: 现有可视化器实例，如果为None则会在initialize时创建
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "Visualizer Plugin"
        self._version = "1.0.0"
        self._description = "将可视化器适配为标准插件"
        self._config = plugin_config or {}

        # 适配的可视化器实例
        self._visualizer = visualizer

        # 插件状态
        self._initialized = False
        self._enabled = False

        logger.info(f"可视化器适配器已创建: {plugin_id}")

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
        return []  # 没有特定依赖

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

            # 如果没有可视化器实例，尝试创建一个
            if self._visualizer is None:
                from visualizer import EnhancedVisualizer
                room_width = self._config.get('room_width', 400)
                room_height = self._config.get('room_height', 300)
                trail_length = self._config.get('trail_length', 50)

                self._visualizer = EnhancedVisualizer(
                    room_width=room_width,
                    room_height=room_height,
                    trail_length=trail_length,
                    config=self._config
                )
                logger.info(
                    f"已创建新的可视化器实例: {room_width}x{room_height}")

            # 添加额外的初始化逻辑
            if context:
                # 例如，如果上下文中有显示管理器，可以设置关联
                display_manager = context.get('display_manager')
                if display_manager:
                    logger.info(f"将可视化器关联到显示管理器")
                    display_manager.set_visualizer(self._visualizer)

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

            # 如果已经初始化，应用配置到可视化器
            if self._visualizer and hasattr(self._visualizer, '_apply_config'):
                self._visualizer._apply_config(config)
                logger.info(f"插件 {self._id} 配置已更新并应用")
            else:
                logger.info(f"插件 {self._id} 配置已更新，将在初始化时应用")

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

            # 可视化器通常不需要特殊清理
            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 实现VisualizerPluginInterface特定方法 =============

    def visualize_frame(self, frame, person=None, action=None, detector=None):
        """
        可视化相机帧

        Args:
            frame: 相机帧图像
            person: 检测到的人体数据
            action: 识别的动作
            detector: 检测器实例

        Returns:
            ndarray: 带有可视化效果的帧
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法可视化帧")
            return frame

        try:
            return self._visualizer.visualize_frame(frame, person, action,
                                                    detector)
        except Exception as e:
            logger.error(f"可视化帧时出错: {e}")
            return frame

    def visualize_room(self, position=None, depth=None, action=None):
        """
        可视化房间地图

        Args:
            position: 房间中的(x,y)位置
            depth: 深度/距离估计
            action: 识别的动作

        Returns:
            ndarray: 房间可视化图像
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法可视化房间")
            return None

        try:
            return self._visualizer.visualize_room(position, depth, action)
        except Exception as e:
            logger.error(f"可视化房间时出错: {e}")
            return None

    def add_trail_point(self, x, y):
        """
        添加轨迹点

        Args:
            x: X坐标
            y: Y坐标
        """
        if not self.is_enabled():
            return

        try:
            self._visualizer.add_trail_point(x, y)
        except Exception as e:
            logger.error(f"添加轨迹点时出错: {e}")

    def draw_debug_info(self, frame, fps=None, system_state=None,
                        debug_data=None):
        """
        绘制调试信息

        Args:
            frame: 输入帧
            fps: 当前帧率
            system_state: 系统状态
            debug_data: 调试数据

        Returns:
            ndarray: 带调试信息的帧
        """
        if not self.is_enabled():
            return frame

        try:
            return self._visualizer.draw_debug_info(frame, fps, system_state,
                                                    debug_data)
        except Exception as e:
            logger.error(f"绘制调试信息时出错: {e}")
            return frame

    def draw_occlusion_message(self, frame):
        """
        绘制遮挡信息

        Args:
            frame: 输入帧

        Returns:
            ndarray: 带遮挡信息的帧
        """
        if not self.is_enabled():
            return frame

        try:
            return self._visualizer.draw_occlusion_message(frame)
        except Exception as e:
            logger.error(f"绘制遮挡信息时出错: {e}")
            return frame

    def get_visualizer(self):
        """
        获取原始可视化器实例

        Returns:
            EnhancedVisualizer: 可视化器实例
        """
        return self._visualizer

    def set_visualizer(self, visualizer):
        """
        设置可视化器实例

        Args:
            visualizer: 新的可视化器实例
        """
        self._visualizer = visualizer
        logger.info(f"已更新可视化器实例")

    def get_debug_info(self):
        """
        获取调试信息

        Returns:
            dict: 调试信息
        """
        if not self.is_enabled() or not self._visualizer:
            return {}

        try:
            debug_info = self._visualizer.get_debug_info()
            # 添加插件相关信息
            debug_info.update({
                'plugin_id': self._id,
                'plugin_version': self._version,
                'plugin_enabled': self._enabled
            })
            return debug_info
        except Exception as e:
            logger.error(f"获取调试信息时出错: {e}")
            return {
                'error': str(e),
                'plugin_id': self._id,
                'plugin_enabled': self._enabled
            }


# 插件系统工厂方法
def create_plugin(plugin_id="visualizer_plugin", config=None,
                  context=None) -> PluginInterface:
    """
    创建可视化器插件实例

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
        plugin = VisualizerAdapter(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建可视化器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建可视化器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
