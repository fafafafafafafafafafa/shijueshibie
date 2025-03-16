# -*- coding: utf-8 -*-
"""
映射器适配器模块 - 将现有的位置映射器类适配为插件系统中的插件

此适配器允许位置映射器作为标准插件集成到插件系统中，遵循插件接口规范，
同时保留其所有原始功能。
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

# 导入基础插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    MapperPluginInterface
)

# 日志配置
from utils.logger_config import get_logger

logger = get_logger("MapperAdapter")


class MapperAdapter(MapperPluginInterface):
    """
    映射器适配器类，将位置映射器适配为符合MapperPluginInterface的插件

    此适配器遵循适配器设计模式，封装现有映射器组件并提供统一的插件接口。
    """

    def __init__(self, mapper=None, plugin_id="mapper_plugin",
                 plugin_config=None):
        """
        初始化映射器适配器

        Args:
            mapper: 现有映射器实例，如果为None则会在initialize时创建
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "Mapper Plugin"
        self._version = "1.0.0"
        self._description = "将位置映射器适配为标准插件"
        self._config = plugin_config or {}

        # 适配的映射器实例
        self._mapper = mapper

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 事件系统引用
        self._event_system = None

        logger.info(f"映射器适配器已创建: {plugin_id}")

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
        return []  # 默认没有依赖

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

            # 如果没有映射器实例，尝试创建一个
            if self._mapper is None:
                from position_mapper import PositionMapper

                camera_height = self._config.get('camera_height', 2.0)
                room_width = self._config.get('room_width', 400)
                room_height = self._config.get('room_height', 300)

                self._mapper = PositionMapper(
                    camera_height=camera_height,
                    room_width=room_width,
                    room_height=room_height
                )
                logger.info(f"已创建新的映射器实例")

            # 从上下文中获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']

                # 如果映射器支持事件系统，设置它
                if hasattr(self._mapper, 'set_event_system'):
                    self._mapper.set_event_system(self._event_system)
                    logger.info(f"已设置映射器的事件系统")

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

            # 如果已经初始化，应用配置到映射器
            if self._mapper and hasattr(self._mapper, 'configure'):
                self._mapper.configure(config)
                logger.info(f"插件 {self._id} 配置已更新并应用")
            elif self._mapper and hasattr(self._mapper, 'update_params'):
                # 提取映射器可能需要的参数
                params = {}
                if 'camera_height' in config:
                    params['camera_height'] = config['camera_height']
                if 'room_width' in config:
                    params['room_width'] = config['room_width']
                if 'room_height' in config:
                    params['room_height'] = config['room_height']
                if 'calibration' in config:
                    params['calibration'] = config['calibration']

                if params:
                    self._mapper.update_params(**params)
                    logger.info(f"插件 {self._id} 参数已更新")
            else:
                logger.info(f"插件 {self._id} 配置已更新，将在重新初始化时应用")

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

            # 调用映射器的清理方法（如果有）
            if self._mapper and hasattr(self._mapper, 'cleanup'):
                self._mapper.cleanup()

            # 清除引用
            self._event_system = None

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 实现MapperPluginInterface特定方法 =============

    def map_position(self, detection_data: Dict[str, Any]) -> Tuple[
        Optional[float], Optional[float], Optional[float]]:
        """
        将检测数据映射到房间位置

        Args:
            detection_data: 检测数据，包含关键点、边界框等

        Returns:
            Tuple: (x, y, depth) 房间坐标和深度，如果映射失败则返回None
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法映射位置")
            return None, None, None

        try:
            # 调用映射器的位置映射方法
            if hasattr(self._mapper, 'map_position'):
                return self._mapper.map_position(detection_data)
            else:
                logger.error(f"映射器没有实现map_position方法")
                return None, None, None
        except Exception as e:
            logger.error(f"映射位置时出错: {e}")
            return None, None, None

    def calibrate(self, calibration_data: Dict[str, Any] = None) -> bool:
        """
        校准映射器

        Args:
            calibration_data: 校准数据

        Returns:
            bool: 校准是否成功
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法校准")
            return False

        try:
            # 调用映射器的校准方法
            if hasattr(self._mapper, 'calibrate'):
                result = self._mapper.calibrate(calibration_data)
                if result:
                    logger.info(f"映射器校准成功")
                else:
                    logger.warning(f"映射器校准未完成")
                return result
            else:
                logger.error(f"映射器没有实现calibrate方法")
                return False
        except Exception as e:
            logger.error(f"校准映射器时出错: {e}")
            return False

    def get_room_dimensions(self) -> Tuple[int, int]:
        """
        获取房间尺寸

        Returns:
            Tuple: (宽度, 高度) 虚拟房间的像素尺寸
        """
        if not self._mapper:
            return (
                self._config.get('room_width', 400),
                self._config.get('room_height', 300)
            )

        try:
            # 如果映射器有获取尺寸的方法，直接调用
            if hasattr(self._mapper, 'get_room_dimensions'):
                return self._mapper.get_room_dimensions()

            # 否则尝试从属性中获取
            if hasattr(self._mapper, 'room_width') and hasattr(self._mapper,
                                                               'room_height'):
                return (self._mapper.room_width, self._mapper.room_height)

            # 兜底使用配置中的值
            return (
                self._config.get('room_width', 400),
                self._config.get('room_height', 300)
            )
        except Exception as e:
            logger.error(f"获取房间尺寸时出错: {e}")
            return (400, 300)  # 默认值

    def get_calibration_status(self) -> Dict[str, Any]:
        """
        获取校准状态

        Returns:
            Dict: 校准状态信息
        """
        if not self.is_enabled() or not self._mapper:
            return {'calibrated': False}

        try:
            # 如果映射器有获取校准状态的方法，直接调用
            if hasattr(self._mapper, 'get_calibration_status'):
                return self._mapper.get_calibration_status()

            # 否则尝试从属性中获取
            if hasattr(self._mapper, 'is_calibrated'):
                is_calibrated = self._mapper.is_calibrated
                return {'calibrated': is_calibrated}

            # 兜底返回未知状态
            return {'calibrated': False, 'status': 'unknown'}
        except Exception as e:
            logger.error(f"获取校准状态时出错: {e}")
            return {'calibrated': False, 'error': str(e)}

    def reset(self) -> bool:
        """
        重置映射器状态

        Returns:
            bool: 重置是否成功
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法重置")
            return False

        try:
            # 调用映射器的重置方法（如果有）
            if hasattr(self._mapper, 'reset'):
                self._mapper.reset()
                logger.info(f"映射器已重置")
                return True
            else:
                logger.warning(f"映射器没有实现reset方法，无法重置")
                return False
        except Exception as e:
            logger.error(f"重置映射器时出错: {e}")
            return False

    def get_mapper(self):
        """
        获取原始映射器实例

        Returns:
            原始映射器实例
        """
        return self._mapper

    def set_mapper(self, mapper):
        """
        设置映射器实例

        Args:
            mapper: 新的映射器实例
        """
        self._mapper = mapper

        # 如果已设置事件系统，同时设置给新映射器
        if self._event_system and hasattr(mapper, 'set_event_system'):
            mapper.set_event_system(self._event_system)

        logger.info(f"已更新映射器实例")


# 插件系统工厂方法
def create_plugin(plugin_id="mapper_plugin", config=None,
                  context=None) -> PluginInterface:
    """
    创建映射器插件实例

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
        plugin = MapperAdapter(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建映射器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建映射器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
