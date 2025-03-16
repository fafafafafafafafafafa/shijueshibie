# -*- coding: utf-8 -*-
"""
高级映射器插件模块 - 提供增强的位置映射功能

此模块提供了一个高级位置映射器插件，支持更精确的人体位置映射、复杂的校准算法
以及深度估计等高级功能。
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
import math
import json
import os

# 导入插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    MapperPluginInterface
)

# 导入日志配置
from utils.logger_config import get_logger

logger = get_logger("AdvancedMapperPlugin")


class AdvancedMapperPlugin(MapperPluginInterface):
    """
    高级映射器插件 - 提供增强的位置映射功能

    此插件使用高级算法将检测到的人体位置映射到虚拟房间坐标系统，
    支持深度估计、多点校准和鲁棒性优化。
    """

    def __init__(self, plugin_id="advanced_mapper", plugin_config=None):
        """
        初始化高级映射器插件

        Args:
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "Advanced Mapper"
        self._version = "1.0.0"
        self._description = "提供增强的位置映射功能的插件"
        self._config = plugin_config or {}

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 映射器相关属性
        self._camera_height = self._config.get('camera_height', 2.0)  # 相机高度(米)
        self._camera_angle = self._config.get('camera_angle', 45.0)  # 相机倾斜角度(度)
        self._room_width = self._config.get('room_width', 500)  # 虚拟房间宽度(像素)
        self._room_height = self._config.get('room_height', 400)  # 虚拟房间高度(像素)
        self._scale_factor = self._config.get('scale_factor',
                                              100.0)  # 缩放因子(像素/米)

        # 校准数据
        self._calibration_points = []  # 校准点 [(image_x, image_y, room_x, room_y, depth), ...]
        self._calibration_matrix = None  # 校准变换矩阵
        self._is_calibrated = False
        self._calibration_file = self._config.get('calibration_file',
                                                  'config/calibration.json')

        # 位置平滑设置
        self._smoothing_enabled = self._config.get('smoothing_enabled', True)
        self._smoothing_factor = self._config.get('smoothing_factor',
                                                  0.5)  # 平滑因子(0-1)
        self._position_history = []  # 历史位置 [(x, y, depth), ...]
        self._history_length = self._config.get('history_length', 5)  # 历史长度

        # 深度估计设置
        self._depth_estimation_method = self._config.get('depth_estimation',
                                                         'height_based')
        self._depth_scale = self._config.get('depth_scale', 1.0)  # 深度比例因子
        self._min_depth = self._config.get('min_depth', 0.5)  # 最小深度(米)
        self._max_depth = self._config.get('max_depth', 10.0)  # 最大深度(米)

        # 异常检测和筛选
        self._outlier_threshold = self._config.get('outlier_threshold',
                                                   1.5)  # 异常阈值(标准差)
        self._max_jump_distance = self._config.get('max_jump_distance',
                                                   50.0)  # 最大跳跃距离(像素)

        # 事件系统引用
        self._event_system = None

        # 统计信息
        self._mapping_count = 0
        self._last_mapping_time = 0
        self._average_mapping_time = 0

        logger.info(f"高级映射器插件已创建: {plugin_id}")

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
        return []  # 高级映射器通常不依赖其他插件

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

            # 尝试加载校准数据
            self._load_calibration()

            # 获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']
                logger.info("已设置事件系统")

                # 发布初始化事件
                if hasattr(self._event_system, 'publish'):
                    self._event_system.publish(
                        "plugin_initialized",
                        {
                            'plugin_id': self._id,
                            'plugin_type': 'mapper',
                            'is_calibrated': self._is_calibrated
                        }
                    )

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

            # 发布启用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_enabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'mapper'
                    }
                )

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

            # 发布禁用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_disabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'mapper'
                    }
                )

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

            # 更新映射器参数
            if 'camera_height' in config:
                self._camera_height = config['camera_height']

            if 'camera_angle' in config:
                self._camera_angle = config['camera_angle']

            if 'room_width' in config:
                self._room_width = config['room_width']

            if 'room_height' in config:
                self._room_height = config['room_height']

            if 'scale_factor' in config:
                self._scale_factor = config['scale_factor']

            if 'smoothing_enabled' in config:
                self._smoothing_enabled = config['smoothing_enabled']

            if 'smoothing_factor' in config:
                self._smoothing_factor = config['smoothing_factor']

            if 'history_length' in config:
                self._history_length = config['history_length']
                # 调整历史记录长度
                if len(self._position_history) > self._history_length:
                    self._position_history = self._position_history[
                                             -self._history_length:]

            if 'depth_estimation' in config:
                self._depth_estimation_method = config['depth_estimation']

            if 'depth_scale' in config:
                self._depth_scale = config['depth_scale']

            if 'outlier_threshold' in config:
                self._outlier_threshold = config['outlier_threshold']

            if 'max_jump_distance' in config:
                self._max_jump_distance = config['max_jump_distance']

            if 'calibration_file' in config:
                self._calibration_file = config['calibration_file']
                # 重新加载校准数据
                self._load_calibration()

            # 记录日志
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

            # 清理资源
            self._position_history.clear()

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
            # 测量映射时间
            start_time = time.time()

            # 从检测数据中提取必要信息
            keypoints = detection_data.get('keypoints', [])
            bbox = detection_data.get('bbox', None)

            if not keypoints and not bbox:
                logger.warning("检测数据中缺少关键点和边界框信息")
                return None, None, None

            # 计算位置
            if self._is_calibrated and self._calibration_matrix is not None:
                # 使用校准矩阵进行高级映射
                position = self._map_with_calibration(keypoints, bbox)
            else:
                # 使用基本几何方法进行映射
                position = self._map_with_geometry(keypoints, bbox)

            if position is None:
                return None, None, None

            # 应用平滑处理
            if self._smoothing_enabled and self._position_history:
                position = self._apply_smoothing(position)

            # 检测并过滤异常值
            if self._position_history and self._is_outlier(position):
                # 使用最后一个有效位置
                position = self._position_history[-1]
                logger.debug("检测到异常值，使用历史位置")

            # 记录位置历史
            self._add_to_history(position)

            # 更新统计信息
            mapping_time = time.time() - start_time
            self._update_stats(mapping_time)

            # 发布位置映射事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "position_mapped",
                    {
                        'position': (position[0], position[1]),
                        'depth': position[2],
                        'mapping_time': mapping_time,
                        'plugin_id': self._id
                    }
                )

            return position

        except Exception as e:
            logger.error(f"映射位置时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None, None

    def calibrate(self, calibration_data: Dict[str, Any] = None) -> bool:
        """
        校准映射器

        Args:
            calibration_data: 校准数据，包含参考点等

        Returns:
            bool: 校准是否成功
        """
        try:
            if not calibration_data:
                # 如果没有提供校准数据，重置校准状态
                self._is_calibrated = False
                self._calibration_matrix = None
                self._calibration_points = []
                logger.info("校准已重置")
                return True

            # 提取校准点
            points = calibration_data.get('points', [])
            if not points or len(points) < 3:
                logger.error("校准点不足，需要至少3个点")
                return False

            # 添加新的校准点
            for point in points:
                if len(point) >= 5:  # (image_x, image_y, room_x, room_y, depth)
                    self._calibration_points.append(point)

            # 确保至少有3个校准点
            if len(self._calibration_points) < 3:
                logger.error("有效校准点不足，需要至少3个点")
                return False

            # 计算校准矩阵
            success = self._compute_calibration_matrix()

            if success:
                # 保存校准数据
                self._save_calibration()

                # 发布校准完成事件
                if self._event_system and hasattr(self._event_system,
                                                  'publish'):
                    self._event_system.publish(
                        "calibration_completed",
                        {
                            'plugin_id': self._id,
                            'points_count': len(self._calibration_points)
                        }
                    )

                logger.info(
                    f"校准成功完成，使用 {len(self._calibration_points)} 个点")
                return True
            else:
                logger.error("计算校准矩阵失败")
                return False

        except Exception as e:
            logger.error(f"校准映射器时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def get_room_dimensions(self) -> Tuple[int, int]:
        """
        获取房间尺寸

        Returns:
            Tuple: (宽度, 高度) 虚拟房间的像素尺寸
        """
        return (self._room_width, self._room_height)

    def get_calibration_status(self) -> Dict[str, Any]:
        """
        获取校准状态

        Returns:
            Dict: 校准状态信息
        """
        return {
            'calibrated': self._is_calibrated,
            'points_count': len(self._calibration_points),
            'method': 'matrix' if self._calibration_matrix is not None else 'geometry'
        }

    def reset(self) -> bool:
        """
        重置映射器状态

        Returns:
            bool: 重置是否成功
        """
        try:
            # 清空位置历史
            self._position_history.clear()

            logger.info("映射器状态已重置")
            return True
        except Exception as e:
            logger.error(f"重置映射器时出错: {e}")
            return False

    # ============= 辅助方法 =============

    def _map_with_calibration(self, keypoints, bbox):
        """
        使用校准矩阵进行映射

        Args:
            keypoints: 关键点数据
            bbox: 边界框数据

        Returns:
            Tuple: (x, y, depth) 映射后的位置和深度
        """
        try:
            # 确定人体位置点（使用关键点或边界框）
            image_point = self._get_human_position(keypoints, bbox)
            if image_point is None:
                return None

            # 使用校准矩阵进行变换
            image_x, image_y = image_point

            # 创建齐次坐标 [x, y, 1]
            homogeneous_point = np.array([image_x, image_y, 1.0])

            # 应用变换矩阵
            transformed = np.dot(self._calibration_matrix, homogeneous_point)

            # 归一化
            if transformed[2] != 0:
                room_x = transformed[0] / transformed[2]
                room_y = transformed[1] / transformed[2]
            else:
                room_x = transformed[0]
                room_y = transformed[1]

            # 估计深度
            depth = self._estimate_depth(keypoints, bbox)

            # 确保坐标在房间范围内
            room_x = max(0, min(room_x, self._room_width))
            room_y = max(0, min(room_y, self._room_height))

            return (room_x, room_y, depth)

        except Exception as e:
            logger.error(f"使用校准矩阵映射位置时出错: {e}")
            return None

    def _map_with_geometry(self, keypoints, bbox):
        """
        使用基本几何方法进行映射

        Args:
            keypoints: 关键点数据
            bbox: 边界框数据

        Returns:
            Tuple: (x, y, depth) 映射后的位置和深度
        """
        try:
            # 确定人体位置点（使用关键点或边界框）
            image_point = self._get_human_position(keypoints, bbox)
            if image_point is None:
                return None

            # 获取图像尺寸（假设图像中心为参考点）
            # 这里通常需要知道相机的分辨率，这里使用一个估计值
            img_width = 640  # 假设宽度
            img_height = 480  # 假设高度

            # 归一化图像坐标 (-1 到 1 范围)
            norm_x = (image_point[0] - img_width / 2) / (img_width / 2)
            norm_y = (image_point[1] - img_height / 2) / (img_height / 2)

            # 应用相机角度和高度调整
            angle_rad = math.radians(self._camera_angle)
            depth_factor = math.cos(angle_rad)  # 深度投影因子

            # 估计深度
            depth = self._estimate_depth(keypoints, bbox)

            # 计算房间坐标
            room_center_x = self._room_width / 2
            room_center_y = self._room_height / 2

            # X坐标与图像水平位置成比例
            room_x = room_center_x + norm_x * depth * self._scale_factor

            # Y坐标与图像垂直位置和深度有关
            room_y = room_center_y + norm_y * depth * self._scale_factor * depth_factor

            # 确保坐标在房间范围内
            room_x = max(0, min(room_x, self._room_width))
            room_y = max(0, min(room_y, self._room_height))

            return (room_x, room_y, depth)

        except Exception as e:
            logger.error(f"使用几何方法映射位置时出错: {e}")
            return None

    def _get_human_position(self, keypoints, bbox):
        """
        确定人体在图像中的位置点

        Args:
            keypoints: 关键点数据
            bbox: 边界框数据

        Returns:
            Tuple: (x, y) 图像中的位置
        """
        # 首选使用关键点
        if keypoints and len(keypoints) > 0:
            # 查找躯干中心点 (使用肩部和髋部关键点的平均值)
            torso_points = []

            # 肩部关键点 (通常索引为5和6)
            if len(keypoints) > 6:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                if len(left_shoulder) >= 2 and len(right_shoulder) >= 2 and \
                        left_shoulder[2] > 0.2 and right_shoulder[2] > 0.2:
                    torso_points.append((left_shoulder[0], left_shoulder[1]))
                    torso_points.append((right_shoulder[0], right_shoulder[1]))

            # 髋部关键点 (通常索引为11和12)
            if len(keypoints) > 12:
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                if len(left_hip) >= 2 and len(right_hip) >= 2 and left_hip[
                    2] > 0.2 and right_hip[2] > 0.2:
                    torso_points.append((left_hip[0], left_hip[1]))
                    torso_points.append((right_hip[0], right_hip[1]))

            # 如果找到躯干点，计算中心
            if torso_points:
                x_sum = sum(p[0] for p in torso_points)
                y_sum = sum(p[1] for p in torso_points)
                return (x_sum / len(torso_points), y_sum / len(torso_points))

        # 如果关键点不可用或不足，使用边界框
        if bbox and len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            # 使用边界框底部中心作为人体位置
            return (x + w / 2, y + h)

        return None

    def _estimate_depth(self, keypoints, bbox):
        """
        估计人体深度

        Args:
            keypoints: 关键点数据
            bbox: 边界框数据

        Returns:
            float: 估计的深度(米)
        """
        if self._depth_estimation_method == 'height_based':
            # 使用人体高度(边界框)估计深度
            if bbox and len(bbox) >= 4:
                _, _, _, height = bbox[:4]
                # 身高与深度成反比
                # 假设标准高度为200像素对应2米深度
                standard_height = 200
                standard_depth = 2.0

                if height > 0:
                    depth = standard_depth * (
                                standard_height / height) * self._depth_scale
                    return max(self._min_depth, min(depth, self._max_depth))

        elif self._depth_estimation_method == 'keypoint_based':
            # 使用关键点估计深度
            if keypoints and len(keypoints) > 0:
                # 计算关键点之间的距离(如肩宽)
                shoulder_width = 0

                # 肩部关键点 (通常索引为5和6)
                if len(keypoints) > 6:
                    left_shoulder = keypoints[5]
                    right_shoulder = keypoints[6]
                    if len(left_shoulder) >= 2 and len(right_shoulder) >= 2 and \
                            left_shoulder[2] > 0.2 and right_shoulder[2] > 0.2:
                        shoulder_width = math.sqrt(
                            (left_shoulder[0] - right_shoulder[0]) ** 2 +
                            (left_shoulder[1] - right_shoulder[1]) ** 2)

                if shoulder_width > 0:
                    # 假设标准肩宽为60像素对应2米深度
                    standard_width = 60
                    standard_depth = 2.0

                    depth = standard_depth * (
                                standard_width / shoulder_width) * self._depth_scale
                    return max(self._min_depth, min(depth, self._max_depth))

        # 默认深度
        return 2.0 * self._depth_scale

    def _apply_smoothing(self, position):
        """
        应用位置平滑处理

        Args:
            position: 当前位置 (x, y, depth)

        Returns:
            Tuple: 平滑后的位置
        """
        if not self._position_history:
            return position

        # 获取最后一个位置
        last_pos = self._position_history[-1]

        # 线性插值平滑
        smooth_x = last_pos[0] * (1 - self._smoothing_factor) + position[
            0] * self._smoothing_factor
        smooth_y = last_pos[1] * (1 - self._smoothing_factor) + position[
            1] * self._smoothing_factor
        smooth_depth = last_pos[2] * (1 - self._smoothing_factor) + position[
            2] * self._smoothing_factor

        return (smooth_x, smooth_y, smooth_depth)

    def _add_to_history(self, position):
        """
        添加位置到历史记录

        Args:
            position: 当前位置 (x, y, depth)
        """
        self._position_history.append(position)

        # 限制历史记录长度
        if len(self._position_history) > self._history_length:
            self._position_history = self._position_history[
                                     -self._history_length:]

    def _is_outlier(self, position):
        """
        检测位置是否为异常值

        Args:
            position: 当前位置 (x, y, depth)

        Returns:
            bool: 是否为异常值
        """
        if len(self._position_history) < 2:
            return False

        # 获取最后一个位置
        last_pos = self._position_history[-1]

        # 计算距离跳变
        dist = math.sqrt(
            (position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2)

        # 如果距离超过阈值，判定为异常
        if dist > self._max_jump_distance:
            logger.debug(
                f"检测到异常跳变: {dist:.2f} > {self._max_jump_distance:.2f}")
            return True

        # 如果历史足够长，计算统计异常
        if len(self._position_history) >= 5:
            # 计算过去几帧的位置统计
            x_vals = [p[0] for p in self._position_history[-5:]]
            y_vals = [p[1] for p in self._position_history[-5:]]

            x_mean = sum(x_vals) / len(x_vals)
            y_mean = sum(y_vals) / len(y_vals)

            x_std = math.sqrt(
                sum((x - x_mean) ** 2 for x in x_vals) / len(x_vals))
            y_std = math.sqrt(
                sum((y - y_mean) ** 2 for y in y_vals) / len(y_vals))

            # 如果当前位置偏离均值超过标准差的倍数，判定为异常
            x_diff = abs(position[0] - x_mean) / (x_std if x_std > 0 else 1)
            y_diff = abs(position[1] - y_mean) / (y_std if y_std > 0 else 1)

            if x_diff > self._outlier_threshold or y_diff > self._outlier_threshold:
                logger.debug(
                    f"检测到统计异常: x_diff={x_diff:.2f}, y_diff={y_diff:.2f}, threshold={self._outlier_threshold:.2f}")
                return True

        return False

    def _compute_calibration_matrix(self):
        """
        计算校准变换矩阵

        Returns:
            bool: 计算是否成功
        """
        try:
            # 确保有足够的校准点
            if len(self._calibration_points) < 3:
                logger.error("校准点不足，需要至少3个点")
                return False

            # 准备源点和目标点
            src_points = []  # 图像坐标
            dst_points = []  # 房间坐标

            for point in self._calibration_points:
                src_points.append([point[0], point[1]])
                dst_points.append([point[2], point[3]])

            # 转换为NumPy数组
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)

            # 计算单应性矩阵 (透视变换)
            if len(src_points) >= 4:
                # 使用RANSAC算法提高鲁棒性
                self._calibration_matrix, _ = cv2.findHomography(src_points,
                                                                 dst_points,
                                                                 cv2.RANSAC,
                                                                 5.0)
            else:
                # 点数不足时使用仿射变换
                self._calibration_matrix = cv2.getAffineTransform(
                    src_points[:3], dst_points[:3])
                # 扩展为3x3矩阵
                self._calibration_matrix = np.vstack(
                    [self._calibration_matrix, [0, 0, 1]])

            # 验证矩阵是否有效
            if self._calibration_matrix is None or np.isnan(
                    self._calibration_matrix).any():
                logger.error("计算得到的校准矩阵无效")
                self._calibration_matrix = None
                self._is_calibrated = False
                return False

            self._is_calibrated = True
            logger.info(f"校准矩阵计算成功:\n{self._calibration_matrix}")
            return True

        except Exception as e:
            logger.error(f"计算校准矩阵时出错: {e}")
            self._calibration_matrix = None
            self._is_calibrated = False
            return False

    def _load_calibration(self):
        """
        从文件加载校准数据

        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(self._calibration_file):
                logger.warning(f"校准文件不存在: {self._calibration_file}")
                return False

            with open(self._calibration_file, 'r') as f:
                data = json.load(f)

            if 'calibration_points' not in data or 'calibration_matrix' not in data:
                logger.error("校准文件格式无效")
                return False

            self._calibration_points = data['calibration_points']

            # 将矩阵数据转换为NumPy数组
            matrix_data = data['calibration_matrix']
            if matrix_data:
                self._calibration_matrix = np.array(matrix_data)
                self._is_calibrated = True
                logger.info(
                    f"已从文件加载校准数据，包含 {len(self._calibration_points)} 个点")
                return True
            else:
                logger.warning("校准文件中矩阵数据为空")
                return False

        except Exception as e:
            logger.error(f"加载校准数据时出错: {e}")
            return False

    def _save_calibration(self):
        """
        保存校准数据到文件

        Returns:
            bool: 保存是否成功
        """
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(self._calibration_file), exist_ok=True)

            # 准备要保存的数据
            data = {
                'calibration_points': self._calibration_points,
                'calibration_matrix': self._calibration_matrix.tolist() if self._calibration_matrix is not None else None
            }

            with open(self._calibration_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"校准数据已保存到: {self._calibration_file}")
            return True

        except Exception as e:
            logger.error(f"保存校准数据时出错: {e}")
            return False

    def _update_stats(self, mapping_time):
        """更新映射统计信息"""
        self._mapping_count += 1
        self._last_mapping_time = mapping_time

        # 更新平均映射时间
        if self._mapping_count == 1:
            self._average_mapping_time = mapping_time
        else:
            # 移动平均
            alpha = 0.1  # 平滑因子
            self._average_mapping_time = (
                                                     1 - alpha) * self._average_mapping_time + alpha * mapping_time


# 插件系统工厂方法
def create_plugin(plugin_id="advanced_mapper", config=None,
                  context=None) -> PluginInterface:
    """
    创建高级映射器插件实例

    此函数是插件系统识别和加载插件的入口点

    Args:
        plugin_id: 插件唯一标识符
        config: 插件配置
        context: 插件上下文

    Returns:
        PluginInterface: 插件实例
    """
    try:
        # 创建插件实例
        plugin = AdvancedMapperPlugin(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建高级映射器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建高级映射器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
