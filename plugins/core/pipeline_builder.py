# -*- coding: utf-8 -*-
"""
管道构建器 - 负责构建处理管道
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

from .plugin_interface import PluginBase, PluginType
from .plugin_registry import get_plugin_by_id, create_plugin

logger = logging.getLogger("PipelineBuilder")


class Pipeline:
    """
    处理管道类 - 按顺序执行处理步骤
    """

    def __init__(self, name: str, description: str = ""):
        """
        初始化处理管道

        Args:
            name: 管道名称
            description: 管道描述
        """
        self.name = name
        self.description = description
        self.steps = []  # [(step_name, plugin_instance, config), ...]
        self.current_step = 0
        self.input_data = None
        self.output_data = None

    def add_step(self, step_name: str, plugin: PluginBase,
                 config: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加处理步骤

        Args:
            step_name: 步骤名称
            plugin: 插件实例
            config: 步骤配置

        Returns:
            bool: 是否成功添加
        """
        self.steps.append((step_name, plugin, config or {}))
        return True

    def process(self, input_data: Any) -> Any:
        """
        处理输入数据

        Args:
            input_data: 输入数据

        Returns:
            Any: 处理结果
        """
        self.input_data = input_data
        data = input_data
        self.current_step = 0

        # 按顺序执行每个步骤
        for step_name, plugin, config in self.steps:
            try:
                logger.debug(f"执行管道步骤: {step_name}")
                self.current_step += 1

                # 调用插件处理方法
                # 不同类型的插件有不同的处理方法，这里根据插件类型调用相应的方法
                plugin_info = plugin.get_info()
                plugin_type = plugin_info.type

                if plugin_type == PluginType.DETECTOR:
                    data = plugin.detect_pose(data)
                elif plugin_type == PluginType.RECOGNIZER and data:
                    # 对于识别器，如果检测到人，则进行识别
                    if isinstance(data, list) and len(data) > 0:
                        # 处理第一个检测结果
                        person = data[0]
                        action = plugin.recognize_action(person)
                        # 将动作添加到结果中
                        data = {'person': person, 'action': action}
                    else:
                        data = {'person': None, 'action': None}
                elif plugin_type == PluginType.MAPPER and data and isinstance(
                        data, dict):
                    # 对于映射器，如果有人体数据，则进行位置映射
                    person = data.get('person')
                    if person:
                        # 假设配置中有房间尺寸
                        frame_width = config.get('frame_width', 640)
                        frame_height = config.get('frame_height', 480)
                        room_width = config.get('room_width', 800)
                        room_height = config.get('room_height', 600)

                        position = plugin.map_position_to_room(
                            frame_width, frame_height,
                            room_width, room_height,
                            person
                        )
                        # 将位置添加到结果中
                        data['position'] = position[:2]  # x, y
                        data['depth'] = position[2]  # depth
                elif plugin_type == PluginType.VISUALIZER:
                    # 对于可视化器，根据数据类型调用不同方法
                    if isinstance(data, dict):
                        frame = data.get('frame')
                        person = data.get('person')
                        action = data.get('action')
                        position = data.get('position')
                        depth = data.get('depth')

                        if frame is not None:
                            frame_viz = plugin.visualize_frame(frame, person,
                                                               action)
                            data['frame_viz'] = frame_viz

                        if position is not None:
                            room_viz = plugin.visualize_room(position, depth,
                                                             action)
                            data['room_viz'] = room_viz

                # 其他类型的插件或工具类插件，使用通用处理方法
                elif hasattr(plugin, 'process'):
                    data = plugin.process(data, config)
            except Exception as e:
                logger.error(f"执行管道步骤 {step_name} 时出错: {e}")
                # 发生错误但继续处理
                continue

        self.output_data = data
        return data


def build_pipeline(pipeline_config: Dict[str, Any]) -> Optional[Pipeline]:
    """
    根据配置构建处理管道

    Args:
        pipeline_config: 管道配置

    Returns:
        Optional[Pipeline]: 构建的管道，如果构建失败则返回None
    """
    try:
        # 获取管道基本信息
        name = pipeline_config.get('name', 'Default Pipeline')
        description = pipeline_config.get('description', '')

        # 创建管道
        pipeline = Pipeline(name, description)

        # 添加处理步骤
        steps_config = pipeline_config.get('steps', [])
        for step_config in steps_config:
            step_name = step_config.get('name', 'Unnamed Step')
            plugin_id = step_config.get('plugin_id')
            plugin_config = step_config.get('config', {})

            if not plugin_id:
                logger.error(f"步骤 {step_name} 缺少插件ID")
                continue

            # 获取或创建插件实例
            plugin = get_plugin_by_id(plugin_id)
            if not plugin:
                logger.error(f"步骤 {step_name} 的插件 {plugin_id} 不存在")
                continue

            # 添加步骤
            pipeline.add_step(step_name, plugin, plugin_config)

        logger.info(f"已构建管道 {name}，包含 {len(pipeline.steps)} 个步骤")
        return pipeline
    except Exception as e:
        logger.error(f"构建管道失败: {e}")
        return None


def validate_pipeline(pipeline_config: Dict[str, Any]) -> Tuple[
    bool, Optional[str]]:
    """
    验证管道配置

    Args:
        pipeline_config: 管道配置

    Returns:
        Tuple[bool, Optional[str]]: (是否有效, 错误信息)
    """
    try:
        # 检查必需的字段
        if 'steps' not in pipeline_config:
            return False, "缺少必需的 'steps' 字段"

        steps = pipeline_config.get('steps', [])
        if not isinstance(steps, list):
            return False, "'steps' 字段应为数组"

        # 验证每个步骤
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return False, f"步骤 {i + 1} 应为对象"

            if 'plugin_id' not in step:
                return False, f"步骤 {i + 1} 缺少必需的 'plugin_id' 字段"

        # 检查管道中是否包含必要的插件类型
        plugin_types = set()
        for step in steps:
            plugin_id = step.get('plugin_id')
            plugin_info = get_plugin_info(plugin_id)
            if plugin_info:
                plugin_types.add(plugin_info.type)

        # 检查是否缺少关键插件类型
        required_types = {PluginType.DETECTOR, PluginType.RECOGNIZER,
                          PluginType.MAPPER}
        missing_types = required_types - plugin_types

        if missing_types:
            missing_type_names = [t.value for t in missing_types]
            return False, f"管道缺少必要的插件类型: {', '.join(missing_type_names)}"

        return True, None
    except Exception as e:
        logger.error(f"验证管道配置失败: {e}")
        return False, f"验证失败: {e}"
