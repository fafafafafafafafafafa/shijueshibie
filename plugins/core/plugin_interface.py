# -*- coding: utf-8 -*-
"""
插件接口定义 - 所有插件必须实现的基本接口
"""

import abc
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("PluginInterface")


class PluginType(Enum):
    """插件类型枚举"""
    DETECTOR = "detector"  # 检测器插件
    RECOGNIZER = "recognizer"  # 识别器插件
    MAPPER = "mapper"  # 位置映射器插件
    VISUALIZER = "visualizer"  # 可视化器插件
    TOOL = "tool"  # 工具类插件
    OTHER = "other"  # 其他类型插件


@dataclass
class PluginInfo:
    """插件信息数据类"""
    id: str  # 插件唯一标识
    name: str  # 插件名称
    version: str = "1.0.0"  # 插件版本
    type: PluginType = PluginType.OTHER  # 插件类型
    description: str = ""  # 插件描述
    author: str = ""  # 插件作者
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他插件
    tags: Set[str] = field(default_factory=set)  # 插件标签
    config_schema: Dict[str, Any] = field(default_factory=dict)  # 配置模式
    capabilities: Set[str] = field(default_factory=set)  # 插件能力

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'type': self.type.value,
            'description': self.description,
            'author': self.author,
            'dependencies': self.dependencies,
            'tags': list(self.tags),
            'capabilities': list(self.capabilities)
        }
        if self.config_schema:
            result['config_schema'] = self.config_schema
        return result


class PluginBase(abc.ABC):
    """
    插件基础接口 - 所有插件必须实现
    """

    @abc.abstractmethod
    def get_info(self) -> PluginInfo:
        """
        获取插件信息

        Returns:
            PluginInfo: 插件信息对象
        """
        pass

    @abc.abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化插件

        Args:
            config: 插件配置

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abc.abstractmethod
    def shutdown(self) -> bool:
        """
        关闭插件并清理资源

        Returns:
            bool: 关闭是否成功
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> Tuple[
        bool, Optional[str]]:
        """
        验证插件配置

        Args:
            config: 要验证的配置

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        # 基本实现，子类可以重写
        return True, None

    def get_capabilities(self) -> Set[str]:
        """
        获取插件支持的功能

        Returns:
            Set[str]: 功能集合
        """
        return self.get_info().capabilities

    def has_capability(self, capability: str) -> bool:
        """
        检查插件是否支持指定功能

        Args:
            capability: 功能名称

        Returns:
            bool: 是否支持
        """
        return capability in self.get_capabilities()


class DetectorPlugin(PluginBase):
    """
    检测器插件接口 - 用于人体姿态检测
    """

    @abc.abstractmethod
    def detect_pose(self, frame) -> List[Dict[str, Any]]:
        """
        检测人体姿态

        Args:
            frame: 输入图像帧

        Returns:
            List[Dict[str, Any]]: 检测到的人体姿态列表
        """
        pass

    def set_detection_params(self, params: Dict[str, Any]) -> bool:
        """
        设置检测参数

        Args:
            params: 检测参数

        Returns:
            bool: 设置是否成功
        """
        # 子类可以重写此方法
        return True


class RecognizerPlugin(PluginBase):
    """
    识别器插件接口 - 用于动作识别
    """

    @abc.abstractmethod
    def recognize_action(self, pose_data: Dict[str, Any]) -> str:
        """
        识别动作

        Args:
            pose_data: 姿态数据

        Returns:
            str: 识别的动作名称
        """
        pass

    def get_supported_actions(self) -> List[str]:
        """
        获取支持的动作类型

        Returns:
            List[str]: 支持的动作类型列表
        """
        # 子类应重写此方法，默认返回空列表
        return []


class MapperPlugin(PluginBase):
    """
    映射器插件接口 - 用于位置映射
    """

    @abc.abstractmethod
    def map_position_to_room(self,
                             frame_width: int,
                             frame_height: int,
                             room_width: int,
                             room_height: int,
                             pose_data: Dict[str, Any]) -> Tuple[
        float, float, float]:
        """
        将检测到的位置映射到房间坐标系

        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
            room_width: 房间宽度
            room_height: 房间高度
            pose_data: 姿态数据

        Returns:
            Tuple[float, float, float]: 映射后的坐标 (x, y, depth)
        """
        pass


class VisualizerPlugin(PluginBase):
    """
    可视化器插件接口 - 用于可视化展示
    """

    @abc.abstractmethod
    def visualize_frame(self, frame, person: Optional[Dict[str, Any]] = None,
                        action: Optional[str] = None,
                        detector=None) -> Any:
        """
        可视化相机帧

        Args:
            frame: 输入图像帧
            person: 人体姿态数据
            action: 动作名称
            detector: 检测器实例

        Returns:
            Any: 可视化后的帧
        """
        pass

    @abc.abstractmethod
    def visualize_room(self, position: Optional[Tuple[float, float]] = None,
                       depth: Optional[float] = None,
                       action: Optional[str] = None) -> Any:
        """
        可视化房间视图

        Args:
            position: 位置坐标
            depth: 深度值
            action: 动作名称

        Returns:
            Any: 可视化后的房间视图
        """
        pass


class ToolPlugin(PluginBase):
    """
    工具类插件接口 - 用于提供辅助功能
    """

    @abc.abstractmethod
    def execute(self, command: str,
                params: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行工具命令

        Args:
            command: 命令名称
            params: 命令参数

        Returns:
            Any: 命令执行结果
        """
        pass

    def get_commands(self) -> List[str]:
        """
        获取支持的命令列表

        Returns:
            List[str]: 命令列表
        """
        # 子类应重写此方法，默认返回空列表
        return []
