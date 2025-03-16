# -*- coding: utf-8 -*-
"""
MediaPipe检测器插件模块 - 基于MediaPipe的人体检测与姿态估计插件

此模块提供了一个基于Google MediaPipe框架的人体检测和姿态估计插件，
能够检测图像中的人体并提供精确的关键点信息。
"""

import logging
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

# 导入插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    DetectorPluginInterface
)

# 导入日志配置
from utils.logger_config import get_logger

logger = get_logger("MediaPipeDetectorPlugin")


class MediaPipeDetectorPlugin(DetectorPluginInterface):
    """
    MediaPipe检测器插件 - 基于MediaPipe的人体检测和姿态估计插件

    此插件使用Google的MediaPipe框架来检测图像中的人体，
    并提供精确的姿态估计和关键点信息。
    """

    def __init__(self, plugin_id="mediapipe_detector", plugin_config=None):
        """
        初始化MediaPipe检测器插件

        Args:
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "MediaPipe Detector"
        self._version = "1.0.0"
        self._description = "基于MediaPipe框架的人体检测和姿态估计插件"
        self._config = plugin_config or {}

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 检测器相关属性
        self._detector = None
        self._min_detection_confidence = self._config.get(
            'min_detection_confidence', 0.5)
        self._min_tracking_confidence = self._config.get(
            'min_tracking_confidence', 0.5)
        self._model_complexity = self._config.get('model_complexity', 1)
        self._static_image_mode = self._config.get('static_image_mode', False)

        # 检测统计
        self._detection_count = 0
        self._last_detection_time = 0
        self._average_detection_time = 0

        # 缓存设置
        self._cache_enabled = self._config.get('cache_enabled', True)
        self._cache_skip_frames = self._config.get('cache_skip_frames', 2)
        self._last_frame = None
        self._last_result = None
        self._frame_count = 0

        # 事件系统引用
        self._event_system = None

        # 镜像设置
        self._mirror_image = self._config.get('mirror_image', False)

        logger.info(f"MediaPipe检测器插件已创建: {plugin_id}")

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
        return []  # MediaPipe检测器不依赖其他插件

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

            # 导入MediaPipe
            try:
                import mediapipe as mp
                self.mp = mp
            except ImportError:
                logger.error(
                    "未安装MediaPipe，请使用 'pip install mediapipe' 安装")
                return False

    # ============= 实现DetectorPluginInterface特定方法 =============

    def detect_person(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        在图像中检测人体并进行姿态估计

        Args:
            frame: 输入图像帧

        Returns:
            Dict: 包含检测结果的字典，如果没有检测到则返回None
            {
                'bbox': [x, y, width, height],
                'keypoints': [[x1, y1, conf1], [x2, y2, conf2], ...],
                'confidence': float  # 检测置信度
            }
        """
        if not self.is_enabled() or self._detector is None:
            logger.warning(f"插件 {self._id} 未启用或未初始化，无法检测")
            return None

        try:
            # 缓存逻辑 - 如果启用了缓存并且不是关键帧，返回缓存的结果
            if self._cache_enabled and self._frame_count % self._cache_skip_frames != 0:
                self._frame_count += 1
                if self._last_result is not None:
                    return self._last_result

            self._frame_count += 1

            # 测量检测时间
            start_time = time.time()

            if frame is None or frame.size == 0:
                logger.warning("检测输入帧无效")
                return None

            # 获取检测器
            pose = self._detector['pose']

            # MediaPipe需要RGB图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 如果启用镜像，进行水平翻转
            if self._mirror_image:
                frame_rgb = cv2.flip(frame_rgb, 1)

            # 执行姿态估计
            results = pose.process(frame_rgb)

            # 如果没有检测到人体，返回None
            if not results.pose_landmarks:
                self._last_result = None
                return None

            # 获取图像尺寸
            height, width = frame.shape[:2]

            # 提取关键点
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0, 0

            # 转换为所需格式并计算边界框
            for landmark in landmarks:
                x, y = int(landmark.x * width), int(landmark.y * height)
                visibility = landmark.visibility if hasattr(landmark,
                                                            'visibility') else 1.0

                keypoints.append([x, y, visibility])

                # 更新边界框坐标
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

            # 计算边界框
            bbox_x = max(0, min_x - 20)
            bbox_y = max(0, min_y - 20)
            bbox_width = min(width - bbox_x, max_x - min_x + 40)
            bbox_height = min(height - bbox_y, max_y - min_y + 40)

            bbox = [bbox_x, bbox_y, bbox_width, bbox_height]

            # 计算整体置信度 (使用可见关键点的平均可见度)
            confidence = sum(kp[2] for kp in keypoints) / len(keypoints)

            # 创建结果
            result = {
                'bbox': bbox,
                'keypoints': keypoints,
                'confidence': confidence,
                'detector': 'mediapipe',
                'pose_landmarks': results.pose_landmarks,  # 保留原始地标数据供可视化使用
                'pose_world_landmarks': results.pose_world_landmarks  # 3D地标
            }

            # 更新统计信息
            detection_time = time.time() - start_time
            self._update_stats(detection_time)

            # 缓存结果
            self._last_frame = frame.copy() if not self._mirror_image else cv2.flip(
                frame.copy(), 1)
            self._last_result = result

            # 发布检测事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                # 创建一个不包含原始地标数据的副本以避免序列化问题
                event_result = result.copy()
                event_result.pop('pose_landmarks', None)
                event_result.pop('pose_world_landmarks', None)

                self._event_system.publish(
                    "person_detected",
                    {
                        'person': event_result,
                        'detection_time': detection_time,
                        'plugin_id': self._id
                    }
                )

            return result

        except Exception as e:
            logger.error(f"检测人体时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _update_stats(self, detection_time):
        """更新检测统计信息"""
        self._detection_count += 1
        self._last_detection_time = detection_time

        # 更新平均检测时间
        if self._detection_count == 1:
            self._average_detection_time = detection_time
        else:
            # 移动平均
            alpha = 0.1  # 平滑因子
            self._average_detection_time = (
                                                       1 - alpha) * self._average_detection_time + alpha * detection_time

    def get_detector_info(self) -> Dict[str, Any]:
        """
        获取检测器信息

        Returns:
            Dict: 检测器信息
        """
        return {
            'detector_type': 'mediapipe',
            'model_complexity': self._model_complexity,
            'min_detection_confidence': self._min_detection_confidence,
            'min_tracking_confidence': self._min_tracking_confidence,
            'static_image_mode': self._static_image_mode,
            'detection_count': self._detection_count,
            'average_detection_time': self._average_detection_time,
            'last_detection_time': self._last_detection_time,
            'mirror_image': self._mirror_image,
            'initialized': self._initialized,
            'enabled': self._enabled
        }


# 插件系统工厂方法
def create_plugin(plugin_id="mediapipe_detector", config=None,
                  context=None) -> PluginInterface:
    """
    创建MediaPipe检测器插件实例

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
        plugin = MediaPipeDetectorPlugin(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建MediaPipe检测器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建MediaPipe检测器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

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

            # 检查是否需要重新初始化检测器
            need_reinit = False

            # 更新检测器参数
            if 'min_detection_confidence' in config:
                self._min_detection_confidence = config[
                    'min_detection_confidence']
                need_reinit = True

            if 'min_tracking_confidence' in config:
                self._min_tracking_confidence = config[
                    'min_tracking_confidence']
                need_reinit = True

            if 'model_complexity' in config:
                self._model_complexity = config['model_complexity']
                need_reinit = True

            if 'static_image_mode' in config:
                self._static_image_mode = config['static_image_mode']
                need_reinit = True

            if 'cache_enabled' in config:
                self._cache_enabled = config['cache_enabled']

            if 'cache_skip_frames' in config:
                self._cache_skip_frames = config['cache_skip_frames']

            if 'mirror_image' in config:
                self._mirror_image = config['mirror_image']

            # 如果需要重新初始化检测器，且已经初始化过
            if need_reinit and self._initialized and self._detector:
                logger.info("配置已更改，需要重新初始化检测器")

                # 清理旧的检测器
                if 'pose' in self._detector:
                    self._detector['pose'].close()

                # 创建新的检测器
                mp_pose = self._detector['mp_pose']
                pose = mp_pose.Pose(
                    static_image_mode=self._static_image_mode,
                    model_complexity=self._model_complexity,
                    min_detection_confidence=self._min_detection_confidence,
                    min_tracking_confidence=self._min_tracking_confidence
                )

                self._detector['pose'] = pose
                logger.info("检测器已重新初始化")

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

            # 清理检测器
            if self._detector and 'pose' in self._detector:
                self._detector['pose'].close()

            self._detector = None
            self._last_frame = None
            self._last_result = None

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

            # 加载MediaPipe姿态估计模型
            logger.info(f"正在加载MediaPipe姿态估计模型...")
            start_time = time.time()

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=self._static_image_mode,
                model_complexity=self._model_complexity,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence
            )

            self._detector = {
                'pose': pose,
                'mp_pose': mp_pose,
                'mp_drawing': mp.solutions.drawing_utils
            }

            load_time = time.time() - start_time
            logger.info(f"MediaPipe模型加载完成，耗时: {load_time:.2f}秒")

            # 获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']
                logger.info("已设置事件系统")

                # 发布模型加载完成事件
                if hasattr(self._event_system, 'publish'):
                    self._event_system.publish(
                        "model_loaded",
                        {
                            'model_type': 'mediapipe',
                            'load_time': load_time,
                            'plugin_id': self._id
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
                        'plugin_type': 'detector',
                        'detector_type': 'mediapipe'
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
                        'plugin_type': 'detector',
                        'detector_type': 'mediapipe'
                    }
                )

            return True
        except Exception as e:
            logger.error(f"禁用插件 {self._id} 时出错: {e}")
            return False
