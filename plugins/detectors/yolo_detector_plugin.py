# -*- coding: utf-8 -*-
"""
YOLO检测器插件模块 - 基于YOLO的人体检测插件实现

此模块提供了一个基于YOLO深度学习模型的人体检测插件，
能够检测图像中的人体并提供边界框和关键点信息。
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

logger = get_logger("YOLODetectorPlugin")


class YOLODetectorPlugin(DetectorPluginInterface):
    """
    YOLO检测器插件 - 基于YOLO模型的人体检测插件

    此插件使用YOLO（You Only Look Once）深度学习模型来检测图像中的人体，
    并提供边界框和姿态估计信息。
    """

    def __init__(self, plugin_id="yolo_detector", plugin_config=None):
        """
        初始化YOLO检测器插件

        Args:
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "YOLO Detector"
        self._version = "1.0.0"
        self._description = "基于YOLO深度学习模型的人体检测插件"
        self._config = plugin_config or {}

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 检测器相关属性
        self._detector = None
        self._confidence_threshold = self._config.get('confidence_threshold',
                                                      0.5)
        self._nms_threshold = self._config.get('nms_threshold', 0.4)
        self._model_path = self._config.get('model_path',
                                            'models/yolo/yolov4-p5.weights')
        self._config_path = self._config.get('config_path',
                                             'models/yolo/yolov4-p5.cfg')
        self._size = self._config.get('size', (640, 640))

        # 检测统计
        self._detection_count = 0
        self._last_detection_time = 0
        self._average_detection_time = 0

        # 是否使用GPU加速
        self._use_cuda = self._config.get('use_cuda', False)

        # 检测器自信度阈值
        self._person_threshold = self._config.get('person_threshold', 0.5)

        # 检测缓存
        self._last_frame = None
        self._last_result = None
        self._cache_enabled = self._config.get('cache_enabled', True)
        self._cache_skip_frames = self._config.get('cache_skip_frames', 2)
        self._frame_count = 0

        # 事件系统引用
        self._event_system = None

        logger.info(f"YOLO检测器插件已创建: {plugin_id}")

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
        return []  # YOLO检测器不依赖其他插件

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

            # 加载YOLO模型
            logger.info(f"正在加载YOLO模型: {self._model_path}")
            start_time = time.time()

            # 创建神经网络
            network = cv2.dnn.readNetFromDarknet(self._config_path,
                                                 self._model_path)

            # 设置后端和目标设备
            if self._use_cuda:
                logger.info("启用CUDA加速")
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # 获取模型输出层名称
            layer_names = network.getLayerNames()
            try:
                # OpenCV 4.5.4+
                output_layers = [layer_names[i - 1] for i in
                                 network.getUnconnectedOutLayers()]
            except:
                # 旧版本兼容
                output_layers = [layer_names[i[0] - 1] for i in
                                 network.getUnconnectedOutLayers()]

            self._detector = {
                'network': network,
                'output_layers': output_layers,
                'size': self._size
            }

            load_time = time.time() - start_time
            logger.info(f"YOLO模型加载完成，耗时: {load_time:.2f}秒")

            # 获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']
                logger.info("已设置事件系统")

                # 发布模型加载完成事件
                if hasattr(self._event_system, 'publish'):
                    self._event_system.publish(
                        "model_loaded",
                        {
                            'model_type': 'yolo',
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
                        'detector_type': 'yolo'
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
                        'detector_type': 'yolo'
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

            # 更新检测器参数
            if 'confidence_threshold' in config:
                self._confidence_threshold = config['confidence_threshold']

            if 'nms_threshold' in config:
                self._nms_threshold = config['nms_threshold']

            if 'person_threshold' in config:
                self._person_threshold = config['person_threshold']

            if 'cache_enabled' in config:
                self._cache_enabled = config['cache_enabled']

            if 'cache_skip_frames' in config:
                self._cache_skip_frames = config['cache_skip_frames']

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
            self._detector = None
            self._last_frame = None
            self._last_result = None

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 实现DetectorPluginInterface特定方法 =============

    def detect_person(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        在图像中检测人体

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
            # 缓存逻辑 - 如果启用了缓存并且帧与上一帧相似，返回缓存的结果
            if self._cache_enabled and self._frame_count % self._cache_skip_frames != 0:
                self._frame_count += 1
                if self._last_result is not None:
                    return self._last_result

            self._frame_count += 1

            # 测量检测时间
            start_time = time.time()

            # 调整图像大小并创建blob
            network = self._detector['network']
            output_layers = self._detector['output_layers']
            target_size = self._detector['size']

            if frame is None or frame.size == 0:
                logger.warning("检测输入帧无效")
                return None

            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, target_size,
                                         [0, 0, 0], swapRB=True, crop=False)

            # 设置输入并执行前向传播
            network.setInput(blob)
            outputs = network.forward(output_layers)

            # 处理检测结果
            boxes = []
            confidences = []
            class_ids = []

            # 解析输出
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # 只处理人类检测结果（类别0）且置信度高于阈值
                    if class_id == 0 and confidence > self._person_threshold:
                        # YOLO返回边界框的中心(x,y)以及宽度和高度
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        detected_width = int(detection[2] * width)
                        detected_height = int(detection[3] * height)

                        # 计算左上角坐标
                        x = int(center_x - detected_width / 2)
                        y = int(center_y - detected_height / 2)

                        boxes.append([x, y, detected_width, detected_height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # 应用非极大值抑制(NMS)
            indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                       self._confidence_threshold,
                                       self._nms_threshold)

            # 如果没有检测到人，返回None
            if len(indices) == 0:
                self._last_result = None
                return None

            # 获取具有最高置信度的人体检测结果
            max_conf_idx = np.argmax([confidences[i] for i in indices])
            idx = indices[max_conf_idx]

            # OpenCV 4.5.4+与旧版本兼容
            if isinstance(idx, np.ndarray):
                idx = idx[0]
            elif isinstance(indices, np.ndarray) and len(indices.shape) > 1:
                idx = indices[max_conf_idx][0]

            # 提取检测框
            bbox = boxes[idx]
            confidence = confidences[idx]

            # 生成一些简单的关键点 (目前YOLO不直接提供关键点，这是一个简化实现)
            # 在实际应用中，可能需要另一个模型来获取精确的关键点
            keypoints = self._generate_simple_keypoints(bbox)

            # 创建结果
            result = {
                'bbox': bbox,
                'keypoints': keypoints,
                'confidence': confidence,
                'detector': 'yolo'
            }

            # 更新统计信息
            detection_time = time.time() - start_time
            self._update_stats(detection_time)

            # 缓存结果
            self._last_frame = frame.copy()
            self._last_result = result

            # 发布检测事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "person_detected",
                    {
                        'person': result,
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

    def _generate_simple_keypoints(self, bbox):
        """
        从边界框生成简化的关键点

        Args:
            bbox: 边界框 [x, y, width, height]

        Returns:
            List: 简化的关键点列表
        """
        x, y, w, h = bbox

        # 创建17个简化关键点，与COCO格式兼容
        keypoints = [
            # 头部和颈部
            [x + w / 2, y + h / 4, 0.9],  # 0: 鼻子
            [x + w / 2 - w / 8, y + h / 5, 0.8],  # 1: 左眼
            [x + w / 2 + w / 8, y + h / 5, 0.8],  # 2: 右眼
            [x + w / 2 - w / 6, y + h / 4.5, 0.7],  # 3: 左耳
            [x + w / 2 + w / 6, y + h / 4.5, 0.7],  # 4: 右耳

            # 肩膀
            [x + w / 4, y + h / 3, 0.8],  # 5: 左肩
            [x + 3 * w / 4, y + h / 3, 0.8],  # 6: 右肩

            # 肘部
            [x + w / 6, y + h / 2, 0.7],  # 7: 左肘
            [x + 5 * w / 6, y + h / 2, 0.7],  # 8: 右肘

            # 手腕
            [x + w / 8, y + 2 * h / 3, 0.6],  # 9: 左手腕
            [x + 7 * w / 8, y + 2 * h / 3, 0.6],  # 10: 右手腕

            # 臀部
            [x + w / 3, y + 2 * h / 3, 0.8],  # 11: 左臀
            [x + 2 * w / 3, y + 2 * h / 3, 0.8],  # 12: 右臀

            # 膝盖
            [x + w / 3, y + 5 * h / 6, 0.7],  # 13: 左膝
            [x + 2 * w / 3, y + 5 * h / 6, 0.7],  # 14: 右膝

            # 脚踝
            [x + w / 3, y + h, 0.6],  # 15: 左脚踝
            [x + 2 * w / 3, y + h, 0.6]  # 16: 右脚踝
        ]

        return keypoints

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
            'detector_type': 'yolo',
            'model_path': self._model_path,
            'confidence_threshold': self._confidence_threshold,
            'detection_count': self._detection_count,
            'average_detection_time': self._average_detection_time,
            'last_detection_time': self._last_detection_time,
            'use_cuda': self._use_cuda,
            'initialized': self._initialized,
            'enabled': self._enabled
        }


# 插件系统工厂方法
def create_plugin(plugin_id="yolo_detector", config=None,
                  context=None) -> PluginInterface:
    """
    创建YOLO检测器插件实例

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
        plugin = YOLODetectorPlugin(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建YOLO检测器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建YOLO检测器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
