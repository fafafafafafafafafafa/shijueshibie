# -*- coding: utf-8 -*-
"""
机器学习识别器插件模块 - 基于机器学习模型的动作识别插件实现

此模块提供了一个基于机器学习（如TensorFlow/Keras、sklearn等）的动作识别插件，
能够识别人体姿态序列中的动作模式。
"""

import logging
import time
import os
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union, Deque
from collections import deque

# 导入插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    RecognizerPluginInterface
)

# 导入日志配置
from utils.logger_config import get_logger

logger = get_logger("MLRecognizerPlugin")


class MLRecognizerPlugin(RecognizerPluginInterface):
    """
    机器学习识别器插件 - 基于机器学习模型的动作识别

    此插件使用预训练的机器学习模型识别人体动作序列，
    支持多种动作类型的识别和置信度估计。
    """

    def __init__(self, plugin_id="ml_recognizer", plugin_config=None):
        """
        初始化机器学习识别器插件

        Args:
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "Machine Learning Recognizer"
        self._version = "1.0.0"
        self._description = "基于机器学习模型的动作识别插件"
        self._config = plugin_config or {}

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 识别器相关属性
        self._model = None
        self._model_path = self._config.get('model_path',
                                            'models/ml/action_model.h5')
        self._labels_path = self._config.get('labels_path',
                                             'models/ml/action_labels.txt')
        self._threshold = self._config.get('threshold', 0.7)
        self._sequence_length = self._config.get('sequence_length', 30)

        # 识别窗口和状态
        self._pose_sequence = deque(maxlen=self._sequence_length)
        self._action_window = self._config.get('action_window', 5)  # 动作确认窗口
        self._recent_actions = deque(maxlen=self._action_window)  # 最近识别的动作
        self._current_action = "Unknown"

        # 识别统计
        self._recognition_count = 0
        self._last_recognition_time = 0
        self._average_recognition_time = 0

        # 特征提取和处理设置
        self._feature_extraction_method = self._config.get('feature_extraction',
                                                           'keypoints')
        self._preprocess_func = None
        self._normalization = self._config.get('normalization', True)

        # 事件系统引用
        self._event_system = None

        # 动作标签和类别
        self._action_labels = []
        self._num_classes = 0

        # 调试设置
        self._debug = self._config.get('debug', False)

        logger.info(f"机器学习识别器插件已创建: {plugin_id}")

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
        return []  # 机器学习识别器通常不依赖其他插件

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

            # 尝试导入TensorFlow/Keras
            try:
                import tensorflow as tf
                self._tf = tf
                self._backend = 'tensorflow'
                logger.info("使用TensorFlow后端")
            except ImportError:
                logger.warning("未找到TensorFlow，尝试其他后端...")
                try:
                    # 尝试使用scikit-learn作为备选
                    import sklearn
                    import joblib
                    self._sklearn = sklearn
                    self._joblib = joblib
                    self._backend = 'sklearn'
                    logger.info("使用scikit-learn后端")
                except ImportError:
                    logger.error(
                        "无法导入TensorFlow或scikit-learn，请安装其中一个依赖")
                    return False

            # 加载动作标签
            self._load_action_labels()

            # 加载模型
            logger.info(f"正在加载动作识别模型: {self._model_path}")
            start_time = time.time()

            if self._backend == 'tensorflow':
                self._model = self._tf.keras.models.load_model(self._model_path)
                logger.info(f"模型输入形状: {self._model.input_shape}")
                logger.info(f"模型输出形状: {self._model.output_shape}")
            else:
                self._model = self._joblib.load(self._model_path)
                logger.info(f"已加载scikit-learn模型: {type(self._model)}")

            # 设置预处理函数
            self._setup_preprocessing()

            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")

            # 获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']
                logger.info("已设置事件系统")

                # 发布模型加载完成事件
                if hasattr(self._event_system, 'publish'):
                    self._event_system.publish(
                        "model_loaded",
                        {
                            'model_type': f'ml_{self._backend}',
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
                        'plugin_type': 'recognizer',
                        'recognizer_type': 'ml'
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
                        'plugin_type': 'recognizer',
                        'recognizer_type': 'ml'
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

            # 更新识别器参数
            if 'threshold' in config:
                self._threshold = config['threshold']

            if 'action_window' in config:
                self._action_window = config['action_window']
                self._recent_actions = deque(maxlen=self._action_window)

            if 'debug' in config:
                self._debug = config['debug']

            if 'feature_extraction' in config:
                self._feature_extraction_method = config['feature_extraction']
                # 重新设置预处理函数
                self._setup_preprocessing()

            if 'normalization' in config:
                self._normalization = config['normalization']

            # 序列长度变更需要重新初始化序列队列
            if 'sequence_length' in config:
                new_length = config['sequence_length']
                if new_length != self._sequence_length:
                    self._sequence_length = new_length
                    self._pose_sequence = deque(maxlen=self._sequence_length)
                    logger.info(f"姿态序列长度已更新为: {new_length}")

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
            self._model = None
            self._pose_sequence.clear()
            self._recent_actions.clear()

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 实现RecognizerPluginInterface特定方法 =============

    def recognize_action(self, detection_data: Dict[str, Any]) -> Tuple[
        str, float]:
        """
        识别人体动作

        Args:
            detection_data: 检测数据，包含关键点信息

        Returns:
            Tuple[str, float]: (动作名称, 置信度)
        """
        if not self.is_enabled() or self._model is None:
            logger.warning(f"插件 {self._id} 未启用或未初始化，无法识别动作")
            return "Unknown", 0.0

        try:
            # 测量识别时间
            start_time = time.time()

            # 获取关键点数据
            keypoints = detection_data.get('keypoints', [])
            if not keypoints or len(keypoints) == 0:
                logger.warning("无效的关键点数据，无法识别动作")
                return "Unknown", 0.0

            # 处理关键点
            processed_keypoints = self._preprocess_keypoints(keypoints)

            # 将处理后的关键点添加到序列
            self._pose_sequence.append(processed_keypoints)

            # 如果序列长度不足，返回未知
            if len(self._pose_sequence) < self._sequence_length:
                logger.debug(
                    f"姿态序列不足 ({len(self._pose_sequence)}/{self._sequence_length})，等待更多数据")
                return "Unknown", 0.0

            # 准备序列数据
            sequence = np.array(list(self._pose_sequence))

            # 根据后端进行预测
            action, confidence = self._predict_action(sequence)

            # 添加到最近动作
            self._recent_actions.append(action)

            # 使用多数投票确定最终动作
            if len(self._recent_actions) == self._action_window:
                from collections import Counter
                action_counts = Counter(self._recent_actions)
                most_common_action = action_counts.most_common(1)[0]

                # 仅当最常见动作的计数超过窗口的一半时才更改当前动作
                if most_common_action[1] > self._action_window / 2:
                    self._current_action = most_common_action[0]

            # 更新统计信息
            recognition_time = time.time() - start_time
            self._update_stats(recognition_time)

            # 仅当置信度高于阈值时才发布识别事件
            if confidence > self._threshold and self._current_action != "Unknown":
                # 发布识别事件
                if self._event_system and hasattr(self._event_system,
                                                  'publish'):
                    self._event_system.publish(
                        "action_recognized",
                        {
                            'action': self._current_action,
                            'confidence': confidence,
                            'recognition_time': recognition_time,
                            'plugin_id': self._id
                        }
                    )

            return self._current_action, confidence

        except Exception as e:
            logger.error(f"识别动作时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return "Unknown", 0.0

    def get_recognizer_info(self) -> Dict[str, Any]:
        """
        获取识别器信息

        Returns:
            Dict: 识别器信息
        """
        return {
            'recognizer_type': 'ml',
            'backend': self._backend,
            'model_path': self._model_path,
            'action_labels': self._action_labels,
            'threshold': self._threshold,
            'sequence_length': self._sequence_length,
            'current_sequence_length': len(self._pose_sequence),
            'action_window': self._action_window,
            'current_action': self._current_action,
            'recognition_count': self._recognition_count,
            'average_recognition_time': self._average_recognition_time,
            'last_recognition_time': self._last_recognition_time,
            'initialized': self._initialized,
            'enabled': self._enabled
        }

    def get_supported_actions(self) -> List[str]:
        """
        获取支持的动作列表

        Returns:
            List[str]: 支持的动作名称列表
        """
        return self._action_labels

    def reset(self) -> bool:
        """
        重置识别器状态

        Returns:
            bool: 重置是否成功
        """
        try:
            # 清空序列和状态
            self._pose_sequence.clear()
            self._recent_actions.clear()
            self._current_action = "Unknown"

            logger.info("识别器状态已重置")
            return True
        except Exception as e:
            logger.error(f"重置识别器时出错: {e}")
            return False

    # ============= 辅助方法 =============

    def _load_action_labels(self):
        """加载动作标签"""
        try:
            if os.path.exists(self._labels_path):
                with open(self._labels_path, 'r') as f:
                    self._action_labels = [line.strip() for line in
                                           f.readlines()]
                self._num_classes = len(self._action_labels)
                logger.info(
                    f"已加载 {self._num_classes} 个动作标签: {', '.join(self._action_labels)}")
            else:
                # 默认标签
                self._action_labels = ["Walking", "Sitting", "Standing",
                                       "Waving", "Jumping", "Unknown"]
                self._num_classes = len(self._action_labels)
                logger.warning(
                    f"标签文件未找到，使用默认标签: {', '.join(self._action_labels)}")
        except Exception as e:
            logger.error(f"加载动作标签时出错: {e}")
            # 设置一些默认标签
            self._action_labels = ["Unknown", "Walking", "Sitting", "Standing"]
            self._num_classes = len(self._action_labels)

    def _setup_preprocessing(self):
        """设置预处理函数"""
        if self._feature_extraction_method == 'keypoints':
            self._preprocess_func = self._extract_keypoint_features
        elif self._feature_extraction_method == 'angles':
            self._preprocess_func = self._extract_angle_features
        elif self._feature_extraction_method == 'velocity':
            self._preprocess_func = self._extract_velocity_features
        else:
            # 默认使用关键点特征
            self._preprocess_func = self._extract_keypoint_features
            logger.warning(
                f"未知的特征提取方法: {self._feature_extraction_method}，使用默认方法")

    def _preprocess_keypoints(self, keypoints):
        """预处理关键点数据"""
        if self._preprocess_func:
            return self._preprocess_func(keypoints)

        # 默认处理：将关键点扁平化为1D数组
        flat_keypoints = []
        for kp in keypoints:
            if len(kp) >= 2:  # 至少有x,y坐标
                flat_keypoints.extend(kp[:2])  # 仅使用x,y坐标

        # 归一化(可选)
        if self._normalization and flat_keypoints:
            flat_keypoints = self._normalize_features(flat_keypoints)

        return flat_keypoints

    def _extract_keypoint_features(self, keypoints):
        """从关键点提取特征"""
        # 将关键点扁平化为1D数组
        flat_keypoints = []
        for kp in keypoints:
            if len(kp) >= 2:  # 至少有x,y坐标
                flat_keypoints.extend(kp[:2])  # 仅使用x,y坐标

        # 归一化(可选)
        if self._normalization and flat_keypoints:
            flat_keypoints = self._normalize_features(flat_keypoints)

        return flat_keypoints

    def _extract_angle_features(self, keypoints):
        """提取关节角度特征"""
        # 定义要计算角度的关节三元组 (每个三元组定义了一个角度)
        # 格式: (中心关节索引, 第一连接关节索引, 第二连接关节索引)
        angle_triplets = [
            (0, 1, 4),  # 鼻子-左眼-左耳
            (0, 2, 5),  # 鼻子-右眼-右耳
            (5, 6, 7),  # 左肩-左肘-左腕
            (6, 8, 9),  # 右肩-右肘-右腕
            (11, 13, 15),  # 左髋-左膝-左踝
            (12, 14, 16)  # 右髋-右膝-右踝
        ]

        angles = []
        for center, first, second in angle_triplets:
            if center < len(keypoints) and first < len(
                    keypoints) and second < len(keypoints):
                # 获取三个关节的坐标
                center_joint = keypoints[center][:2]  # 仅使用x,y
                first_joint = keypoints[first][:2]
                second_joint = keypoints[second][:2]

                # 计算向量
                v1 = np.array(first_joint) - np.array(center_joint)
                v2 = np.array(second_joint) - np.array(center_joint)

                # 计算角度 (弧度)
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cosine = np.dot(v1, v2) / (
                                np.linalg.norm(v1) * np.linalg.norm(v2))
                    cosine = max(-1, min(cosine, 1))  # 确保在[-1, 1]范围内
                    angle = np.arccos(cosine)
                    angles.append(angle)
                else:
                    angles.append(0)  # 如果有零向量，设为0
            else:
                angles.append(0)  # 如果索引超出范围，设为0

        return angles

    def _extract_velocity_features(self, keypoints):
        """提取速度特征 (需要连续两帧)"""
        # 如果没有足够的历史数据，使用关键点特征
        if len(self._pose_sequence) == 0:
            return self._extract_keypoint_features(keypoints)

        # 获取前一帧的关键点
        prev_keypoints = self._pose_sequence[-1]

        # 计算关键点速度
        velocities = []

        # 如果前一帧是处理过的特征而不是原始关键点，使用关键点特征
        if len(prev_keypoints) != len(keypoints):
            return self._extract_keypoint_features(keypoints)

        # 计算关键点速度
        for i, (curr_kp, prev_kp) in enumerate(zip(keypoints, prev_keypoints)):
            if len(curr_kp) >= 2 and len(prev_kp) >= 2:
                vx = curr_kp[0] - prev_kp[0]
                vy = curr_kp[1] - prev_kp[1]
                velocities.extend([vx, vy])
            else:
                velocities.extend([0, 0])

        return velocities

    def _normalize_features(self, features):
        """归一化特征"""
        if not features:
            return features

        # 转换为NumPy数组
        features = np.array(features)

        # 如果是第一次，初始化特征数组，不做归一化
        if len(self._pose_sequence) == 0:
            return features.tolist()

        # 通过减去均值并除以标准差进行标准化
        mean = np.mean(features)
        std = np.std(features)

        if std > 0:
            normalized = (features - mean) / std
            return normalized.tolist()
        else:
            return features.tolist()

    def _predict_action(self, sequence):
        """使用模型预测动作"""
        if self._backend == 'tensorflow':
            return self._predict_with_tensorflow(sequence)
        else:
            return self._predict_with_sklearn(sequence)

    def _predict_with_tensorflow(self, sequence):
        """使用TensorFlow/Keras模型预测"""
        try:
            # 准备输入数据
            # 根据模型的输入形状调整
            if len(self._model.input_shape) == 3:  # (batch, time_steps, features)
                x = np.expand_dims(sequence, axis=0)  # 添加批次维度
            else:  # 如果模型期望不同的形状，则需要调整
                x = sequence.reshape(1, -1)  # 扁平化为单个批次

            # 执行预测
            prediction = self._model.predict(x)[0]

            # 获取预测结果
            predicted_class_index = np.argmax(prediction)
            confidence = prediction[predicted_class_index]

            # 获取动作标签
            if predicted_class_index < len(self._action_labels):
                action = self._action_labels[predicted_class_index]
            else:
                action = "Unknown"

            # 仅当置信度高于阈值时才返回预测动作
            if confidence < self._threshold:
                action = "Unknown"

            return action, float(confidence)

        except Exception as e:
            logger.error(f"scikit-learn预测时出错: {e}")
            return "Unknown", 0.0

    def _update_stats(self, recognition_time):
        """更新识别统计信息"""
        self._recognition_count += 1
        self._last_recognition_time = recognition_time

        # 更新平均识别时间
        if self._recognition_count == 1:
            self._average_recognition_time = recognition_time
        else:
            # 移动平均
            alpha = 0.1  # 平滑因子
            self._average_recognition_time = (
                                                         1 - alpha) * self._average_recognition_time + alpha * recognition_time


# 插件系统工厂方法
def create_plugin(plugin_id="ml_recognizer", config=None,
                  context=None) -> PluginInterface:
    """
    创建机器学习识别器插件实例

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
        plugin = MLRecognizerPlugin(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建机器学习识别器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建机器学习识别器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
        "

        # 仅当置信度高于阈值时才返回预测动作
        if confidence < self._threshold:
            action = "Unknown"

        return action, float(confidence)

    except Exception as e:
        logger.error(f"TensorFlow预测时出错: {e}")
        return "Unknown", 0.0


def _predict_with_sklearn(self, sequence):
    """使用scikit-learn模型预测"""
    try:
        # 调整数据格式
        x = sequence.reshape(1, -1)  # 扁平化为单个样本

        # 预测类别
        predicted_class = self._model.predict(x)[0]

        # 尝试获取概率
        try:
            probabilities = self._model.predict_proba(x)[0]
            confidence = max(probabilities)
        except:
            # 如果模型不支持概率，使用简单的置信度估计
            confidence = 0.8  # 默认置信度

        # 获取动作标签
        if predicted_class < len(self._action_labels):
            action = self._action_labels[predicted_class]
        else:
            action = "Unknown"

        # 仅当置信度高于阈值时才返回预测动作
        if confidence < self._threshold:
            action = "Unknown"

        return action, float(confidence)

    except Exception as e:
        logger.error(f"scikit-learn预测时出错: {e}")
        return "Unknown", 0.0
