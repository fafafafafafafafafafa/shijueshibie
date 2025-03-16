# -*- coding: utf-8 -*-
"""
简化版的人体检测器实现，使用组合替代多继承
"""
import cv2
import numpy as np
import time
from interface.detector_interface import PersonDetectorInterface
from utils.data_structures import CircularBuffer
from utils.cache_utils import create_standard_cache
from utils.config import AppConfig
import logging
from utils.logger_config import setup_logger


class SimplifiedPersonDetector(PersonDetectorInterface):
    """
    简化的人体检测器类 - 使用组合模式整合各功能模块

    特点:
    1. 使用组合代替多继承，提高代码清晰度
    2. 保留所有原有功能
    3. 更容易跟踪方法调用和错误定位
    4. 集成事件系统，发布关键操作和状态变化的事件
    """

    def __init__(self, use_mediapipe=False, performance_mode='balanced',
                 event_system=None, config=None):
        """初始化简化检测器"""
        self.logger = logging.getLogger("SimplifiedDetector")
        self.logger.info(
            f"初始化SimplifiedDetector: mediapipe={use_mediapipe}, mode={performance_mode}")

        # 保存事件系统
        self.events = event_system

        # 使用配置对象
        if config is None:
            # 如果没有提供配置，创建一个默认的配置
            config = AppConfig()

        # 从配置对象中获取值
        if isinstance(config, AppConfig):
            self.using_mediapipe = config.get('use_mediapipe', use_mediapipe)
            self.performance_mode = config.get('performance_mode',
                                               performance_mode)
            self.downscale_factor = config.get('downscale_factor', 0.6)
            self.keypoint_confidence_threshold = config.get(
                'keypoint_confidence_threshold', 0.5)
        else:
            # 如果配置对象不是AppConfig类型，使用默认值
            self.using_mediapipe = use_mediapipe
            self.performance_mode = performance_mode
            self.downscale_factor = 0.6
            self.keypoint_confidence_threshold = 0.5


        # 图像处理参数
        self.use_downscale = True
        self.roi_enabled = True  # 默认启用ROI，但是会更智能地自动禁用
        self.last_roi = None
        self.roi_padding = 500  # 极大增加基础ROI扩展像素值
        self.roi_padding_factor = 2.0  # 更大的ROI扩展系数
        self.roi_padding_ratio = 1.0  # 100%的ROI扩展比例
        self.roi_min_ratio = 0.1  # 进一步降低ROI最小比例要求
        self.roi_fallback_frames = 0  # 立即失效，一旦失败立刻回到全图检测
        self.roi_miss_counter = 0
        self.roi_recovery_delay = 0  # ROI恢复延迟计数器
        self.roi_detection_history = []  # 记录ROI检测结果历史
        self.roi_success_threshold = 0.7  # ROI模式需要至少70%的成功率才会启用
        self.full_frame_interval = 2.0  # 每2秒强制一次全图检测
        self.full_frame_sequence = 0  # 当前连续全图检测次数
        self.full_frame_timeout = 0  # 全图检测超时

        # 速度检测参数
        self.motion_history = []  # 记录人体移动历史
        self.max_motion_history = 5  # 最大历史记录长度
        self.high_motion_threshold = 50  # 高速移动阈值（像素/帧)

        # 动态确定降采样比例
        self._set_downscale_factor()

        # 关键点置信度阈值 - 动态自适应参数
        self.keypoint_confidence_threshold = 0.5
        self.detection_timeout = 0.5
        self.history_weight = 0.3

        # 初始化YOLO模型
        self._init_yolo_model()

        # 初始化历史记录
        history_len = 10 if self.performance_mode != 'high_speed' else 5
        self.detection_history = CircularBuffer(history_len)
        self.detection_success_history = CircularBuffer(30)
        self.last_detection_time = 0

        # 初始化组件
        self._init_components()

        # 初始化关键点关系模型
        self.keypoint_relations = self._init_keypoint_relations()

        # 初始化LFU缓存 - 用于缓存检测结果
        self.detection_cache = create_standard_cache(
            name="detector",
            capacity=10,
            timeout=0.5,
            persistent=True
        )
        self.cache_hits = 0
        self.cache_misses = 0

        # 发布初始化完成事件
        if self.events:
            self.events.publish("detector_initialized", {
                'mediapipe_enabled': self.using_mediapipe,
                'performance_mode': self.performance_mode,
                'timestamp': time.time()
            })

        # 打印初始化消息
        self.logger.info(
            f"SimplifiedDetector初始化完成: 性能模式={self.performance_mode}, 降采样系数={self.downscale_factor}")

    def _validate_performance_mode(self, mode):
        """验证性能模式是否有效"""
        valid_modes = ['high_quality', 'balanced', 'high_speed']
        if mode not in valid_modes:
            self.logger.warning(f"无效的性能模式 '{mode}'，使用'balanced'替代")
            return 'balanced'
        return mode

    def _set_downscale_factor(self):
        """根据性能模式设置降采样因子"""
        if self.performance_mode == 'high_speed':
            self.downscale_factor = 0.5  # 缩小一半
        elif self.performance_mode == 'balanced':
            self.downscale_factor = 0.6  # 缩小至60%
        else:  # high_quality
            self.downscale_factor = 0.75  # 缩小至75%

    def _init_yolo_model(self):
        """初始化YOLO模型"""
        try:
            from ultralytics import YOLO

            # 根据性能模式选择模型
            model_name = 'yolov8n-pose.pt'  # 默认使用最轻量级模型
            if self.performance_mode == 'high_quality' and not self._is_low_end_device():
                model_name = 'yolov8m-pose.pt'  # 更精确但较慢的模型

            self.yolo_pose = YOLO(model_name)
            self.logger.info(f"YOLO模型 '{model_name}' 加载成功")

            # 发布模型加载成功事件
            if self.events:
                self.events.publish("yolo_model_loaded", {
                    'model_name': model_name,
                    'timestamp': time.time()
                })
        except Exception as e:
            self.logger.error(f"初始化YOLO模型失败: {e}")

            # 发布模型加载失败事件
            if self.events:
                self.events.publish("yolo_model_load_failed", {
                    'error': str(e),
                    'timestamp': time.time()
                })

            raise

    def _is_low_end_device(self):
        """检测是否为低端设备"""
        import platform
        import psutil

        try:
            # 检查CPU核心数
            cpu_count = psutil.cpu_count(logical=False)
            if cpu_count is None:
                cpu_count = psutil.cpu_count(logical=True)

            # 检查可用RAM
            available_ram = psutil.virtual_memory().total / (
                    1024 * 1024 * 1024)  # GB

            # 根据平台检查是否为移动设备
            system = platform.system()
            is_mobile = system == 'Darwin' and platform.machine() == 'arm64'  # Apple Silicon

            # 低端设备标准: 少于4核或低于8GB RAM或移动设备
            return (cpu_count < 4) or (available_ram < 8) or is_mobile
        except:
            # 如果无法确定，假设为低端设备
            return True

    def _init_components(self):
        """初始化组件"""
        # 初始化MediaPipe（如果启用）
        if self.using_mediapipe:
            self._init_mediapipe()

        # 初始化卡尔曼滤波器
        self.kalman_filters = {}
        self.kalman_last_used = {}
        self.kalman_cleanup_counter = 0

        # 初始化背景减除器
        history_size = 300 if self.performance_mode == 'high_speed' else 500
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history_size, varThreshold=16, detectShadows=False)

        # 初始化连接查找表
        self._init_connection_lookup()

    def _init_mediapipe(self):
        """初始化MediaPipe姿态估计"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            # 使用轻量级模型配置
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 使用最轻量级的模型(0而不是2)
                smooth_landmarks=True,  # 增加平滑度
                enable_segmentation=False,
                min_detection_confidence=0.5)
            self.using_mediapipe = True
            self.logger.info("MediaPipe姿态估计（轻量模式）初始化成功")

            # 发布MediaPipe初始化成功事件
            if self.events:
                self.events.publish("mediapipe_initialized", {
                    'model_complexity': 0,
                    'timestamp': time.time()
                })
        except ImportError:
            self.logger.warning("未安装MediaPipe。深度学习增强不可用。")
            self.logger.warning("安装方法: pip install mediapipe")
            self.using_mediapipe = False

            # 发布MediaPipe初始化失败事件
            if self.events:
                self.events.publish("mediapipe_init_failed", {
                    'reason': 'not_installed',
                    'timestamp': time.time()
                })
        except Exception as e:
            self.logger.error(f"初始化MediaPipe失败: {e}")
            self.using_mediapipe = False

            # 发布MediaPipe初始化错误事件
            if self.events:
                self.events.publish("mediapipe_init_failed", {
                    'reason': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                })

    def _init_connection_lookup(self):
        """初始化关键点连接查找表"""
        # 定义关键点连接关系
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 面部和手臂
            (5, 6), (5, 11), (6, 12), (11, 12),  # 躯干
            (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
        ]

        # 创建查找表，使连接查询更高效
        self.connection_lookup = {}
        for i in range(17):  # YOLO姿态有17个关键点
            self.connection_lookup[i] = []

        for p1, p2 in connections:
            self.connection_lookup[p1].append(p2)
            self.connection_lookup[p2].append(p1)

    def _init_keypoint_relations(self):
        """初始化关键点关系模型"""
        # 关键点索引对应: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
        # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
        # 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
        # 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

        # 格式: 目标点: [(参考点1, 参考点2, 比例), ...]
        # 比例: 表示目标点在参考点1->参考点2的向量上的位置 (0=与点1重合, 1=与点2重合, >1=延长线)
        return {
            # 如果右耳缺失,基于右眼和鼻子推断
            4: [(2, 0, 1.5)],
            # 左耳基于左眼和鼻子
            3: [(1, 0, 1.5)],
            # 右肩基于右眼和右髋
            6: [(2, 12, 0.3)],
            # 左肩基于左眼和左髋
            5: [(1, 11, 0.3)],
            # 右肘基于右肩和右手腕
            8: [(6, 10, 0.5)],
            # 左肘基于左肩和左手腕
            7: [(5, 9, 0.5)],
            # 右膝基于右髋和右踝
            14: [(12, 16, 0.5)],
            # 左膝基于左髋和左踝
            13: [(11, 15, 0.5)]
        }

    def adaptive_parameters(self):
        """基于检测成功率动态调整参数"""
        # 计算检测成功率
        if not self.detection_success_history:
            return

        success_rate = sum(self.detection_success_history) / len(
            self.detection_success_history)

        # 发布检测成功率事件
        if self.events:
            self.events.publish("detection_success_rate_updated", {
                'success_rate': success_rate,
                'timestamp': time.time()
            })

        # 根据检测困难程度动态调整参数
        if success_rate < 0.3:  # 检测极其困难 (不到30%成功率)
            # 显著降低置信度阈值
            old_threshold = self.keypoint_confidence_threshold
            self.keypoint_confidence_threshold = max(0.2,
                                                     self.keypoint_confidence_threshold - 0.05)
            # 大幅增加历史数据权重
            old_weight = self.history_weight
            self.history_weight = min(0.9, self.history_weight + 0.05)
            # 显著增加检测超时
            old_timeout = self.detection_timeout
            self.detection_timeout = min(2.0, self.detection_timeout + 0.1)
            # 在检测极其困难时禁用ROI
            if not hasattr(self, 'roi_enabled_original'):
                self.roi_enabled_original = self.roi_enabled
            self.roi_enabled = False
            # 设置长时间的全图检测模式
            self.full_frame_timeout = 20
            self.logger.warning("检测极其困难，禁用ROI跟踪，启用全图检测模式")

            # 发布参数调整事件
            if self.events:
                self.events.publish("detection_parameters_adjusted", {
                    'reason': 'very_difficult_detection',
                    'success_rate': success_rate,
                    'changes': {
                        'confidence_threshold': {'old': old_threshold,
                                                 'new': self.keypoint_confidence_threshold},
                        'history_weight': {'old': old_weight,
                                           'new': self.history_weight},
                        'detection_timeout': {'old': old_timeout,
                                              'new': self.detection_timeout},
                        'roi_enabled': False,
                        'full_frame_timeout': 20
                    },
                    'timestamp': time.time()
                })
        elif success_rate < 0.5:  # 检测困难
            # 降低置信度阈值
            old_threshold = self.keypoint_confidence_threshold
            self.keypoint_confidence_threshold = max(0.25,
                                                     self.keypoint_confidence_threshold - 0.02)
            # 增加历史数据权重
            old_weight = self.history_weight
            self.history_weight = min(0.8, self.history_weight + 0.02)
            # 增加检测超时
            old_timeout = self.detection_timeout
            self.detection_timeout = min(1.5, self.detection_timeout + 0.1)
            # 检测困难但不至于完全禁用ROI
            if hasattr(self, 'roi_padding_factor_original'):
                # 恢复ROI相关参数，但增加扩展系数
                old_factor = self.roi_padding_factor
                self.roi_padding_factor = min(3.0,
                                              self.roi_padding_factor * 1.2)
                self.logger.info(f"增加ROI扩展系数: {self.roi_padding_factor}")
            else:
                # 保存原始扩展系数并增加
                self.roi_padding_factor_original = self.roi_padding_factor
                old_factor = self.roi_padding_factor
                self.roi_padding_factor = min(3.0,
                                              self.roi_padding_factor * 1.2)

            # 确保ROI至少启用
            if hasattr(self,
                       'roi_enabled_original') and self.roi_enabled_original:
                self.roi_enabled = True

            # 发布参数调整事件
            if self.events:
                self.events.publish("detection_parameters_adjusted", {
                    'reason': 'difficult_detection',
                    'success_rate': success_rate,
                    'changes': {
                        'confidence_threshold': {'old': old_threshold,
                                                 'new': self.keypoint_confidence_threshold},
                        'history_weight': {'old': old_weight,
                                           'new': self.history_weight},
                        'detection_timeout': {'old': old_timeout,
                                              'new': self.detection_timeout},
                        'roi_padding_factor': {'old': old_factor,
                                               'new': self.roi_padding_factor}
                    },
                    'timestamp': time.time()
                })
        elif success_rate > 0.8:  # 检测良好
            # 小幅恢复默认参数，但不要太激进
            old_threshold = self.keypoint_confidence_threshold
            self.keypoint_confidence_threshold = min(0.5,
                                                     self.keypoint_confidence_threshold + 0.01)
            old_weight = self.history_weight
            self.history_weight = max(0.3, self.history_weight - 0.01)
            old_timeout = self.detection_timeout
            self.detection_timeout = max(0.5, self.detection_timeout - 0.05)

            roi_changes = {}

            # 检测良好时逐渐恢复ROI参数
            if hasattr(self, 'roi_padding_factor_original'):
                # 逐渐恢复ROI扩展系数
                factor_diff = self.roi_padding_factor - self.roi_padding_factor_original
                if factor_diff > 0.1:  # 还有差距
                    old_factor = self.roi_padding_factor
                    self.roi_padding_factor -= 0.1
                    self.logger.info(
                        f"检测良好，逐渐恢复ROI扩展系数: {self.roi_padding_factor}")
                    roi_changes['roi_padding_factor'] = {'old': old_factor,
                                                         'new': self.roi_padding_factor}
                else:
                    old_factor = self.roi_padding_factor
                    self.roi_padding_factor = self.roi_padding_factor_original
                    roi_changes['roi_padding_factor'] = {'old': old_factor,
                                                         'new': self.roi_padding_factor}

            # 恢复ROI启用状态
            if hasattr(self, 'roi_enabled_original'):
                old_roi_enabled = self.roi_enabled
                self.roi_enabled = self.roi_enabled_original
                roi_changes['roi_enabled'] = {'old': old_roi_enabled,
                                              'new': self.roi_enabled}

            # 发布参数调整事件
            if self.events:
                self.events.publish("detection_parameters_adjusted", {
                    'reason': 'good_detection',
                    'success_rate': success_rate,
                    'changes': {
                        'confidence_threshold': {'old': old_threshold,
                                                 'new': self.keypoint_confidence_threshold},
                        'history_weight': {'old': old_weight,
                                           'new': self.history_weight},
                        'detection_timeout': {'old': old_timeout,
                                              'new': self.detection_timeout},
                        **roi_changes
                    },
                    'timestamp': time.time()
                })

    def _enhance_with_mediapipe(self, image):
        """使用MediaPipe增强姿态估计"""
        if not self.using_mediapipe or not hasattr(self, 'pose'):
            return None

        # 为提高性能，可以先调整图像大小
        scale_factor = 0.5  # 缩小一半
        small_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                 fy=scale_factor)

        try:
            # 转换到RGB
            image_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

            # 处理图像
            results = self.pose.process(image_rgb)

            if results and hasattr(results,
                                   'pose_landmarks') and results.pose_landmarks:
                # 转换到YOLO格式
                keypoints = []
                h, w, _ = image.shape  # 使用原始图像尺寸
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    # 恢复到原始图像尺寸
                    x = int(landmark.x * small_image.shape[1] / scale_factor)
                    y = int(landmark.y * small_image.shape[0] / scale_factor)
                    keypoints.append([x, y, landmark.visibility])

                # 发布MediaPipe关键点增强事件
                if self.events:
                    self.events.publish("mediapipe_enhancement_applied", {
                        'keypoints_count': len(keypoints),
                        'timestamp': time.time()
                    })

                return np.array(keypoints)
        except Exception as e:
            self.logger.error(f"MediaPipe处理错误: {e}")

            # 发布MediaPipe处理错误事件
            if self.events:
                self.events.publish("mediapipe_processing_error", {
                    'error': str(e),
                    'timestamp': time.time()
                })

        return None

    def _combine_with_mediapipe(self, enhanced_keypoints, mediapipe_keypoints):
        """结合YOLO和MediaPipe的关键点"""
        if mediapipe_keypoints is None:
            return enhanced_keypoints

        result = enhanced_keypoints.copy()

        for j, (x, y, conf) in enumerate(enhanced_keypoints):
            if conf < self.keypoint_confidence_threshold and j < len(
                    mediapipe_keypoints):
                mp_x, mp_y, mp_conf = mediapipe_keypoints[j]
                if mp_conf > 0.5:  # 确保MediaPipe的关键点置信度足够高
                    result[j] = [mp_x, mp_y, mp_conf]

        return result

    def _infer_missing_keypoints(self, keypoints):
        """推断丢失的关键点位置基于身体模型"""
        # 复制关键点数据以避免修改原始数据
        enhanced_keypoints = keypoints.copy()

        # 找出有效关键点
        valid_keypoints = {}
        for i, (x, y, conf) in enumerate(keypoints):
            if conf >= self.keypoint_confidence_threshold:
                valid_keypoints[i] = (x, y)

        # 尝试推断丢失的关键点
        for missing_idx, relations in self.keypoint_relations.items():
            if missing_idx not in valid_keypoints:
                for relation in relations:
                    ref1, ref2, ratio = relation
                    if ref1 in valid_keypoints and ref2 in valid_keypoints:
                        x1, y1 = valid_keypoints[ref1]
                        x2, y2 = valid_keypoints[ref2]
                        dx = x2 - x1
                        dy = y2 - y1
                        # 推断位置
                        inferred_x = x1 + dx * ratio
                        inferred_y = y1 + dy * ratio
                        # 更新关键点，但置信度较低
                        enhanced_keypoints[missing_idx] = [inferred_x,
                                                           inferred_y, 0.3]
                        break

        return enhanced_keypoints

    def _apply_kalman_filter(self, keypoints):
        """对所有关键点应用卡尔曼滤波"""
        filtered_keypoints = keypoints.copy()

        for j, (x, y, conf) in enumerate(keypoints):
            if conf >= self.keypoint_confidence_threshold:
                filtered_x, filtered_y = self._kalman_predict_correct(f"kp_{j}",
                                                                      x, y)
                filtered_keypoints[j] = [filtered_x, filtered_y, conf]

        return filtered_keypoints

    def _init_kalman_filter(self, point_id):
        """初始化特定点的卡尔曼滤波器"""
        # 4维状态: x, y, dx, dy
        # 2维测量: x, y
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]],
                                            np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32)
        kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32) * 0.03
        return kalman

    def _kalman_predict_correct(self, point_id, x, y):
        """使用卡尔曼滤波器预测并修正位置"""
        # 检查输入有效性
        if not np.isfinite(x) or not np.isfinite(y):
            return x, y  # 如果输入无效，直接返回原始值

        # 定期清理卡尔曼滤波器，避免内存泄漏
        self.kalman_cleanup_counter += 1
        if self.kalman_cleanup_counter > 1000:  # 每1000次调用清理一次
            # 移除超过5分钟未使用的滤波器
            current_time = time.time()
            to_remove = []
            for k, v in self.kalman_last_used.items():
                if current_time - v > 300:  # 5分钟 = 300秒
                    to_remove.append(k)
            for k in to_remove:
                if k in self.kalman_filters:
                    del self.kalman_filters[k]
                del self.kalman_last_used[k]
            self.kalman_cleanup_counter = 0

        # 记录最后使用时间
        self.kalman_last_used[point_id] = time.time()

        # 初始化滤波器
        if point_id not in self.kalman_filters:
            self.kalman_filters[point_id] = self._init_kalman_filter(point_id)
            self.kalman_filters[point_id].statePre = np.array(
                [[x], [y], [0], [0]], np.float32)
            self.kalman_filters[point_id].statePost = np.array(
                [[x], [y], [0], [0]], np.float32)
            return x, y

        try:
            # 预测
            prediction = self.kalman_filters[point_id].predict()

            # 更新
            measurement = np.array([[x], [y]], np.float32)
            self.kalman_filters[point_id].correct(measurement)

            # 检查预测值有效性
            pred_x, pred_y = float(prediction[0]), float(prediction[1])
            if not np.isfinite(pred_x) or not np.isfinite(pred_y):
                return x, y

            # 加权平均原始测量和预测值
            filtered_x = x * (
                    1 - self.history_weight) + pred_x * self.history_weight
            filtered_y = y * (
                    1 - self.history_weight) + pred_y * self.history_weight

            return filtered_x, filtered_y
        except Exception as e:
            self.logger.error(f"卡尔曼滤波器错误: {e}")
            return x, y

    def _context_aware_tracking(self, frame, last_detection):
        """利用场景上下文改进追踪"""
        if not last_detection:
            return None

        try:
            # 降低分辨率提高性能
            scale = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            # 应用背景减除 - 使用缩小的帧
            fg_mask = self.background_subtractor.apply(small_frame)

            # 降噪 - 优化核大小以适应缩小的分辨率
            kernel_size = max(2, int(5 * scale))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (kernel_size, kernel_size))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # 寻找运动轮廓
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # 找到最靠近上次检测位置的轮廓
            if contours:
                # 缩放上次检测中心点以匹配缩小的帧
                scaled_last_center = (int(last_detection['center_x'] * scale),
                                      int(last_detection['center_y'] * scale))

                best_contour = None
                min_distance = float('inf')
                min_area = max(250, 1000 * scale * scale)  # 根据缩放调整最小面积阈值

                for c in contours:
                    area = cv2.contourArea(c)
                    if area < min_area:
                        continue

                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dist = ((cx - scaled_last_center[0]) ** 2 + (
                                cy - scaled_last_center[1]) ** 2) ** 0.5
                        if dist < min_distance:
                            min_distance = dist
                            best_contour = c

                # 调整距离阈值以适应缩放
                scaled_distance_threshold = 100 * scale

                if best_contour is not None and min_distance < scaled_distance_threshold:
                    x, y, w, h = cv2.boundingRect(best_contour)

                    # 缩放回原始分辨率
                    x, y = int(x / scale), int(y / scale)
                    w, h = int(w / scale), int(h / scale)

                    # 更新检测结果
                    updated_detection = last_detection.copy()
                    updated_detection['bbox'] = (x, y, x + w, y + h)
                    updated_detection['center_x'] = x + w // 2
                    updated_detection['center_y'] = y + h // 2
                    updated_detection['width'] = w
                    updated_detection['height'] = h
                    updated_detection['motion_based'] = True

                    # 发布上下文追踪成功事件
                    if self.events:
                        self.events.publish("context_tracking_success", {
                            'timestamp': time.time(),
                            'bbox': (x, y, x + w, y + h),
                            'center': (x + w // 2, y + h // 2)
                        })

                    return updated_detection

            # 发布上下文追踪失败事件
            if self.events:
                self.events.publish("context_tracking_failed", {
                    'timestamp': time.time()
                })

            return last_detection
        except Exception as e:
            self.logger.error(f"上下文追踪错误: {e}")

            # 发布上下文追踪错误事件
            if self.events:
                self.events.publish("context_tracking_error", {
                    'error': str(e),
                    'timestamp': time.time()
                })

            return last_detection

    def detect_pose(self, frame):
        """检测人体姿态，整合降采样、ROI处理、MediaPipe增强和关键点处理"""
        current_time = time.time()

        # 注意：使用帧的部分数据作为键可能更高效
        frame_hash = hash(str(frame.mean())) + hash(str(current_time))

        # 尝试从缓存获取结果
        cached_result = self.detection_cache.get(frame_hash)
        if cached_result is not None:
            self.cache_hits += 1
            self.logger.info(
                f"使用缓存检测结果 (命中率: {self.cache_hits / (self.cache_hits + self.cache_misses):.2f})")

            # 发布缓存命中事件
            if self.events:
                self.events.publish("detector_cache_hit", {
                    'hit_rate': self.cache_hits / (
                            self.cache_hits + self.cache_misses),
                    'timestamp': time.time()
                })

            return cached_result

        self.cache_misses += 1

        # 发布缓存未命中事件
        if self.events:
            self.events.publish("detector_cache_miss", {
                'hit_rate': self.cache_hits / (
                        self.cache_hits + self.cache_misses),
                'timestamp': time.time()
            })

        processed_frame = frame  # 默认使用原始帧
        scale_factor = 1.0  # 默认缩放比例
        roi_used = False  # 是否使用了ROI
        roi_coords = None  # ROI坐标，用于恢复原始坐标
        restore_coords = lambda x, y: (x, y)  # 默认坐标恢复函数
        frame_height, frame_width = frame.shape[:2]  # 获取原始帧尺寸

        try:
            # 动态调整参数
            self.adaptive_parameters()

            # 处理ROI恢复延迟计数器
            if self.roi_recovery_delay > 0:
                self.roi_recovery_delay -= 1
                self.logger.info(
                    f"ROI恢复延迟: 还剩{self.roi_recovery_delay}帧")

            # 评估移动速度 - 决定是否应该暂时禁用ROI
            using_roi_is_safe = True
            if hasattr(self, 'last_position') and self.motion_history:
                # 计算平均移动速度
                avg_motion = sum(self.motion_history) / len(self.motion_history)
                if avg_motion > self.high_motion_threshold:
                    using_roi_is_safe = False
                    self.logger.info(
                        f"检测到高速移动: {avg_motion:.1f}像素/帧，暂时禁用ROI")

                    # 发布高速移动事件
                    if self.events:
                        self.events.publish("high_motion_detected", {
                            'avg_motion': avg_motion,
                            'threshold': self.high_motion_threshold,
                            'roi_disabled': True,
                            'timestamp': time.time()
                        })

            # 决定是否使用全图检测
            should_use_full_frame = False

            # 1. 检查全图检测超时
            if self.full_frame_timeout > 0:
                should_use_full_frame = True
                self.full_frame_timeout -= 1
                self.logger.info(
                    f"全图检测模式: 还剩{self.full_frame_timeout}帧")

            # 2. 周期性全图检测
            if hasattr(self, 'last_full_frame_time'):
                time_since_last_full = current_time - self.last_full_frame_time
                if time_since_last_full > self.full_frame_interval:
                    should_use_full_frame = True
                    self.full_frame_sequence += 1  # 增加连续全图检测计数
                    if self.full_frame_sequence >= 3:  # 如果连续3次都用全图
                        self.full_frame_sequence = 0  # 重置计数
                        self.last_full_frame_time = current_time  # 重置时间
                    self.logger.info(
                        f"周期性全图检测 ({time_since_last_full:.1f}秒)")
            else:
                self.last_full_frame_time = current_time

            mediapipe_keypoints = None
            if self.using_mediapipe:
                mediapipe_keypoints = self._enhance_with_mediapipe(
                    processed_frame)

                # 发布MediaPipe处理事件
                if self.events and mediapipe_keypoints is not None:
                    self.events.publish("mediapipe_enhancement_applied", {
                        'keypoints_count': len(mediapipe_keypoints),
                        'timestamp': time.time()
                    })

            # 4. 使用YOLO姿态估计模型
            results = self.yolo_pose(processed_frame, conf=0.1)
            detection_success = False

            persons = []
            for result in results:
                if not hasattr(result, 'keypoints') or result.keypoints is None:
                    continue

                keypoints = result.keypoints.data
                boxes = result.boxes.data

                # 确保有关键点检测
                if len(keypoints) == 0:
                    continue

                for i, (kpt, box) in enumerate(zip(keypoints, boxes)):
                    try:
                        # 将检测到的坐标转换回原始坐标系
                        x1, y1, x2, y2, conf, cls = box.cpu().numpy()

                        # 将边界框坐标转换回原始坐标系
                        orig_x1, orig_y1 = restore_coords(x1, y1)
                        orig_x2, orig_y2 = restore_coords(x2, y2)

                        if conf < 0.5:  # 置信度过滤
                            continue

                        # 获取所有关键点及其置信度
                        kpt_array = kpt.cpu().numpy()

                        # 将关键点坐标转换回原始坐标系
                        for j in range(len(kpt_array)):
                            kx, ky, kconf = kpt_array[j]
                            orig_kx, orig_ky = restore_coords(kx, ky)
                            kpt_array[j] = [orig_kx, orig_ky, kconf]

                        # 1. 推断丢失的关键点
                        enhanced_keypoints = self._infer_missing_keypoints(
                            kpt_array)

                        # 2. 使用MediaPipe增强低置信度关键点
                        if mediapipe_keypoints is not None:
                            enhanced_keypoints = self._combine_with_mediapipe(
                                enhanced_keypoints, mediapipe_keypoints)

                        # 3. 应用卡尔曼滤波平滑关键点
                        filtered_keypoints = self._apply_kalman_filter(
                            enhanced_keypoints)

                        # 确保边界框在合理范围内
                        height, width = frame.shape[:2]
                        orig_x1 = max(0, min(width - 1, int(orig_x1)))
                        orig_y1 = max(0, min(height - 1, int(orig_y1)))
                        orig_x2 = max(0, min(width - 1, int(orig_x2)))
                        orig_y2 = max(0, min(height - 1, int(orig_y2)))

                        # 添加人体信息
                        person = {
                            'bbox': (orig_x1, orig_y1, orig_x2, orig_y2),
                            'center_x': int((orig_x1 + orig_x2) / 2),
                            'center_y': int((orig_y1 + orig_y2) / 2),
                            'width': int(orig_x2 - orig_x1),
                            'height': int(orig_y2 - orig_y1),
                            'keypoints': filtered_keypoints,
                            'original_keypoints': kpt_array,
                            'id': i,
                            'detection_time': current_time,
                            'confidence': float(conf),
                            'is_enhanced': True,
                            'processed_with_roi': roi_used,
                            'processed_with_downscale': self.use_downscale
                        }

                        persons.append(person)
                        detection_success = True

                        # 发布人体检测事件
                        if self.events:
                            self.events.publish("person_detected_detailed", {
                                'bbox': (orig_x1, orig_y1, orig_x2, orig_y2),
                                'center': (
                                    person['center_x'], person['center_y']),
                                'size': (person['width'], person['height']),
                                'confidence': float(conf),
                                'keypoints_count': len(filtered_keypoints),
                                'processed_with_roi': roi_used,
                                'timestamp': time.time()
                            })

                        # 如果检测成功，更新ROI区域
                        if self.roi_enabled:
                            # 保存当前检测框作为下一帧的ROI
                            orig_x1, orig_y1, orig_x2, orig_y2 = person['bbox']

                            # 计算中心点
                            center_x = (orig_x1 + orig_x2) // 2
                            center_y = (orig_y1 + orig_y2) // 2

                            # 计算检测框的宽度和高度
                            det_width = orig_x2 - orig_x1
                            det_height = orig_y2 - orig_y1

                            # 使用超级动态扩展系数，根据多种因素计算
                            # 1. 基础扩展: 基于人体大小
                            person_ratio = (det_width * det_height) / (
                                    frame_width * frame_height)

                            # 极小人体(远距离)使用极大扩展，大人体(近距离)使用小扩展
                            if person_ratio < 0.02:  # 极小人体
                                size_factor = 5.0  # 极大扩展
                            elif person_ratio < 0.05:  # 很小的人体
                                size_factor = 4.0  # 很大扩展
                            elif person_ratio < 0.1:  # 小人体
                                size_factor = 3.0  # 大扩展
                            elif person_ratio < 0.2:  # 中等人体
                                size_factor = 2.0  # 中等扩展
                            else:  # 大人体
                                size_factor = 1.5  # 小扩展

                            # 2. 移动速度因子: 基于历史移动速度
                            motion_factor = 1.0  # 默认无额外扩展
                            if hasattr(self,
                                       'motion_history') and self.motion_history:
                                avg_motion = sum(self.motion_history) / len(
                                    self.motion_history)
                                # 快速移动时使用更大扩展
                                if avg_motion > self.high_motion_threshold:
                                    motion_factor = 2.0  # 快速移动，大幅扩展
                                elif avg_motion > self.high_motion_threshold / 2:
                                    motion_factor = 1.5  # 中速移动，中等扩展

                            # 3. 检测稳定性因子: 基于ROI检测历史
                            stability_factor = 1.0  # 默认无额外扩展
                            if len(self.roi_detection_history) >= 10:
                                roi_success_rate = sum(
                                    self.roi_detection_history[-10:]) / 10
                                # 检测不稳定时使用更大扩展
                                if roi_success_rate < 0.5:
                                    stability_factor = 2.0  # 不稳定，大幅扩展
                                elif roi_success_rate < 0.8:
                                    stability_factor = 1.5  # 较不稳定，中等扩展

                            # 4. 帧间距离因子: 如果当前与上一帧距离较大，使用更大扩展
                            distance_factor = 1.0  # 默认无额外扩展
                            if hasattr(self, 'last_position'):
                                dx = center_x - self.last_position[0]
                                dy = center_y - self.last_position[1]
                                distance = (dx ** 2 + dy ** 2) ** 0.5
                                # 根据移动距离动态调整
                                distance_factor = min(3.0,
                                                      max(1.0, distance / 30))

                            # 整合所有因子，计算最终扩展系数 (使用最大因子加平均因子)
                            max_factor = max(size_factor, motion_factor,
                                             stability_factor, distance_factor)
                            avg_factor = (
                                                 size_factor + motion_factor + stability_factor + distance_factor) / 4
                            expansion_factor = (
                                    max_factor * 0.7 + avg_factor * 0.3)

                            # 确保扩展系数至少为2.0，保证大区域ROI
                            expansion_factor = max(2.0, expansion_factor)

                            # 计算扩展后的宽度和高度
                            width = int(det_width * expansion_factor)
                            height = int(det_height * expansion_factor)

                            # 限制扩展后的大小，确保捕获足够大的区域
                            width = max(width,
                                        int(frame_width * 0.3))  # 至少占图像宽度的30%
                            height = max(height,
                                         int(frame_height * 0.3))  # 至少占图像高度的30%

                            # 确保宽高比适合人体形态
                            aspect_ratio = width / height
                            if aspect_ratio < 0.5:  # 太窄
                                width = int(height * 0.5)
                            elif aspect_ratio > 2.0:  # 太宽
                                height = int(width * 0.5)

                            # 根据当前活动计算额外偏移，预测可能的移动方向
                            offset_x, offset_y = 0, 0
                            if hasattr(self, 'last_position'):
                                dx = center_x - self.last_position[0]
                                dy = center_y - self.last_position[1]
                                # 基于移动方向添加额外偏移，预测下一帧位置
                                offset_x = int(dx * 0.3)  # 向移动方向偏移30%
                                offset_y = int(dy * 0.3)

                            # 计算新的边界框，加入方向偏移，并确保不超出图像边界
                            new_x1 = max(0,
                                         center_x - width // 2 + offset_x)
                            new_y1 = max(0,
                                         center_y - height // 2 + offset_y)
                            new_x2 = min(frame_width - 1,
                                         center_x + width // 2 + offset_x)
                            new_y2 = min(frame_height - 1,
                                         center_y + height // 2 + offset_y)

                            # 确保ROI不会太小 - 边界检查
                            if new_x2 - new_x1 < 50 or new_y2 - new_y1 < 50:
                                # ROI太小，使用中心点扩展为图像的40%大小
                                new_width = int(frame_width * 0.4)
                                new_height = int(frame_height * 0.4)
                                new_x1 = max(0, center_x - new_width // 2)
                                new_y1 = max(0, center_y - new_height // 2)
                                new_x2 = min(frame_width - 1,
                                             center_x + new_width // 2)
                                new_y2 = min(frame_height - 1,
                                             center_y + new_height // 2)

                            # 更新ROI
                            self.last_roi = (new_x1, new_y1, new_x2, new_y2)
                            self.roi_miss_counter = 0

                            # 记录用于日志
                            new_width = new_x2 - new_x1
                            new_height = new_y2 - new_y1
                            new_ratio = (new_width * new_height) / (
                                    frame_width * frame_height)

                            # 发布ROI更新事件
                            if self.events:
                                self.events.publish("roi_updated", {
                                    'old_bbox': (
                                        orig_x1, orig_y1, orig_x2, orig_y2),
                                    'new_roi': (
                                    new_x1, new_y1, new_x2, new_y2),
                                    'expansion_factor': expansion_factor,
                                    'size_factor': size_factor,
                                    'motion_factor': motion_factor,
                                    'stability_factor': stability_factor,
                                    'distance_factor': distance_factor,
                                    'offset': (offset_x, offset_y),
                                    'timestamp': time.time()
                                })

                            self.logger.info(
                                f"更新ROI: ({new_x1},{new_y1},{new_x2},{new_y2}), "
                                f"扩展系数: {expansion_factor:.1f}, 图像比例: {new_ratio:.2f}, "
                                f"移动偏移: ({offset_x},{offset_y})")
                    except Exception as e:
                        self.logger.error(f"处理检测结果时出错: {e}")

                        # 发布检测处理错误事件
                        if self.events:
                            self.events.publish("detection_processing_error", {
                                'error': str(e),
                                'timestamp': time.time()
                            })

                    continue



            # 更新检测历史和成功率
            self.detection_success_history.append(1 if detection_success else 0)

            # 更新ROI检测历史
            if roi_used:
                self.roi_detection_history.append(1 if detection_success else 0)
                # 保持历史记录不超过30帧
                if len(self.roi_detection_history) > 30:
                    self.roi_detection_history = self.roi_detection_history[
                                                 -30:]

            if persons:
                self.detection_history.append(persons[0])
                self.last_detection_time = current_time

                # 更新移动速度计算
                person = persons[0]
                current_pos = (person['center_x'], person['center_y'])

                if hasattr(self, 'last_position'):
                    # 计算移动距离
                    dx = current_pos[0] - self.last_position[0]
                    dy = current_pos[1] - self.last_position[1]
                    distance = (dx ** 2 + dy ** 2) ** 0.5

                    # 更新移动历史
                    self.motion_history.append(distance)
                    if len(self.motion_history) > self.max_motion_history:
                        self.motion_history.pop(0)

                    # 如果移动太快，设置全图检测超时
                    if distance > self.high_motion_threshold * 1.5:  # 非常高速移动
                        self.full_frame_timeout = 5  # 5帧全图检测
                        self.logger.info(
                            f"检测到极高速移动 ({distance:.1f} 像素/帧)，启用全图检测模式")

                        # 发布极高速移动事件
                        if self.events:
                            self.events.publish("extreme_motion_detected", {
                                'distance': distance,
                                'threshold': self.high_motion_threshold * 1.5,
                                'full_frame_timeout': 5,
                                'timestamp': time.time()
                            })

                # 更新上一帧位置
                self.last_position = current_pos

                # 每次成功检测都保存一个全图时间戳（如果是全图检测）
                if not roi_used:
                    self.last_full_frame_time = current_time
                    # 重置全图检测序列
                    self.full_frame_sequence = 0

                # 成功检测后重置ROI相关计数器
                self.roi_miss_counter = 0
                self.roi_recovery_delay = 0

                # 在返回结果前缓存结果
                if persons:  # 如果检测到人
                    self.detection_cache.put(frame_hash, persons)

                # 在检测成功时发布事件
                if persons and hasattr(self, 'events') and self.events:
                    for i, person in enumerate(persons):
                        self.events.publish("person_detected", {
                            'person': person,
                            'frame': frame,
                            'timestamp': time.time(),
                            'index': i,
                            'total': len(persons)
                        })

                if persons and hasattr(self, 'events') and self.events:
                    person = persons[0]
                    self.events.publish("person_detected", {
                        'person': person,
                        'confidence': person.get('confidence', 0.5),
                        'frame': frame,
                        'timestamp': time.time()
                    })

                return persons
            else:
                # 如果使用ROI但未检测到人，增加miss计数并立即切换到全图
                if roi_used:
                    self.roi_miss_counter += 1
                    # 如果连续未检测到人，切换到全图检测并延长恢复时间
                    if self.roi_miss_counter >= self.roi_fallback_frames:
                        self.logger.info("ROI跟踪丢失，切换到全图检测")
                        self.last_roi = None  # 清空ROI
                        # 设置更长的恢复延迟，并使用全图检测超时
                        self.roi_recovery_delay = 8  # 8帧后才允许重新启用ROI
                        self.full_frame_timeout = 10  # 10帧全图检测

                        # 发布ROI跟踪丢失事件
                        if self.events:
                            self.events.publish("roi_tracking_lost", {
                                'miss_counter': self.roi_miss_counter,
                                'recovery_delay': 8,
                                'full_frame_timeout': 10,
                                'timestamp': time.time()
                            })

                # 如果全图检测也失败，可能需要增加容忍度
                elif not roi_used and self.last_roi is not None:
                    # 连续多次全图检测失败，可能需要降低检测阈值
                    if hasattr(self, 'full_frame_fails'):
                        self.full_frame_fails += 1
                        if self.full_frame_fails >= 3:
                            # 降低检测阈值
                            backup_threshold = self.keypoint_confidence_threshold
                            self.keypoint_confidence_threshold = max(0.2,
                                                                     self.keypoint_confidence_threshold - 0.1)
                            self.logger.info(
                                f"连续全图检测失败，降低阈值: {backup_threshold} -> {self.keypoint_confidence_threshold}")

                            # 发布降低检测阈值事件
                            if self.events:
                                self.events.publish(
                                    "detection_threshold_lowered", {
                                        'old_threshold': backup_threshold,
                                        'new_threshold': self.keypoint_confidence_threshold,
                                        'timestamp': time.time()
                                    })

                            self.full_frame_fails = 0
                    else:
                        self.full_frame_fails = 1

                # 发布检测失败事件
                if self.events:
                    self.events.publish("person_detection_failed", {
                        'roi_used': roi_used,
                        'roi_miss_counter': self.roi_miss_counter if roi_used else 0,
                        'timestamp': time.time()
                    })

            # 处理检测失败的情况
            # 如果当前检测失败但有历史记录，使用上下文感知追踪
            if current_time - self.last_detection_time < self.detection_timeout and len(
                    self.detection_history) > 0:
                last_person = list(self.detection_history)[-1]
                context_person = self._context_aware_tracking(frame,
                                                              last_person)
                if context_person:
                    context_person['is_history'] = True
                    # 使用上下文追踪的结果更新ROI
                    if self.roi_enabled and 'bbox' in context_person:
                        self.last_roi = context_person['bbox']
                        self.roi_miss_counter = max(0,
                                                    self.roi_miss_counter - 1)  # 减少miss计数

                    # 发布上下文追踪成功事件
                    if self.events:
                        self.events.publish("context_tracking_success", {
                            'timestamp': time.time(),
                            'bbox': context_person.get('bbox')
                        })

                    return [context_person]
                else:
                    # 发布上下文追踪失败事件
                    if self.events:
                        self.events.publish("context_tracking_failed", {
                            'timestamp': time.time()
                        })

            return []

        except Exception as e:
            self.logger.error(f"YOLO检测错误: {e}")
            self.detection_success_history.append(0)

            # 发布YOLO检测错误事件
            if self.events:
                self.events.publish("yolo_detection_error", {
                    'error': str(e),
                    'timestamp': time.time()
                })

            # 如果在ROI模式下出错，增加miss计数器，可能需要切换回全图检测
            if roi_used:
                self.roi_miss_counter += 1
                if self.roi_miss_counter >= self.roi_fallback_frames:
                    self.logger.warning("ROI跟踪错误，切换到全图检测")
                    self.last_roi = None  # 清空ROI
                    self.roi_recovery_delay = 3  # 强制3帧全图检测

                    # 发布ROI跟踪错误事件
                    if self.events:
                        self.events.publish("roi_tracking_error", {
                            'miss_counter': self.roi_miss_counter,
                            'recovery_delay': 3,
                            'timestamp': time.time()
                        })

            return []

    def draw_skeleton(self, frame, keypoints, color=None):
        """
        在图像上绘制骨架

        Args:
            frame: 输入的图像帧
            keypoints: 关键点列表或包含关键点的人体字典
            color: 可选的颜色参数

        Returns:
            ndarray: 带有骨架的图像
        """
        # 检查参数类型，支持多种格式
        if isinstance(keypoints, dict) and 'keypoints' in keypoints:
            keypoints = keypoints['keypoints']

        # 复制输入帧以避免修改原始数据
        frame_viz = frame.copy()

        # 检查关键点是否有效
        if not keypoints or len(keypoints) < 5:
            return frame_viz

        # COCO 模型的关键点连接定义
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 面部和颈部
            (5, 6), (5, 11), (6, 12), (11, 12),  # 躯干
            (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
        ]

        # 设置默认颜色
        keypoint_color = (0, 255, 0) if color is None else color  # 绿色关键点

        # 设置关键点置信度阈值
        threshold = getattr(self, 'keypoint_confidence_threshold', 0.3)

        # 绘制关键点
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3:  # 确保有 x, y, confidence
                x, y, conf = kp
                if conf > threshold:
                    # 根据关键点类型选择不同颜色
                    if i < 5:  # 头部关键点
                        point_color = (0, 0, 255)  # 红色
                    elif i < 11:  # 上半身关键点
                        point_color = (0, 255, 0)  # 绿色
                    else:  # 下半身关键点
                        point_color = (255, 0, 0)  # 蓝色

                    cv2.circle(frame_viz, (int(x), int(y)), 5, point_color, -1)

        # 绘制连接
        for connection in connections:
            p1, p2 = connection
            if p1 < len(keypoints) and p2 < len(keypoints):
                kp1 = keypoints[p1]
                kp2 = keypoints[p2]

                if len(kp1) >= 3 and len(kp2) >= 3 and kp1[2] > threshold and \
                        kp2[2] > threshold:
                    # 选择连接颜色
                    if p1 < 5 or p2 < 5:  # 头部连接
                        line_color = (0, 0, 255)  # 红色
                    elif p1 < 11 and p2 < 11:  # 上半身连接
                        line_color = (0, 255, 0)  # 绿色
                    else:  # 下半身连接
                        line_color = (255, 0, 0)  # 蓝色

                    cv2.line(frame_viz, (int(kp1[0]), int(kp1[1])),
                             (int(kp2[0]), int(kp2[1])), line_color, 2)

        return frame_viz

    def on_config_changed(self, key, old_value, new_value):
        """
        响应配置系统的变更通知

        Args:
            key: 变更的配置键
            old_value: 变更前的值
            new_value: 变更后的值
        """
        self.logger.info(
            f"检测器配置变更: {key} = {new_value} (原值: {old_value})")

        try:
            # 处理各种配置项变更
            if key == "detector.use_mediapipe":
                self.using_mediapipe = new_value
                # 如果启用MediaPipe但未初始化，则初始化
                if new_value and not hasattr(self, 'pose'):
                    self._init_mediapipe()

            elif key == "detector.performance_mode":
                old_mode = self.performance_mode
                self.performance_mode = self._validate_performance_mode(
                    new_value)
                # 如果性能模式发生实质变化，重新设置相关参数
                if old_mode != self.performance_mode:
                    self._set_downscale_factor()

            elif key == "detector.downscale_factor":
                self.downscale_factor = float(new_value)

            elif key == "detector.keypoint_confidence_threshold":
                self.keypoint_confidence_threshold = float(new_value)

            elif key == "detector.roi_enabled":
                self.roi_enabled = bool(new_value)

            elif key == "detector.roi_padding":
                self.roi_padding = int(new_value)

            elif key == "detector.roi_padding_factor":
                self.roi_padding_factor = float(new_value)

            elif key == "detector.roi_padding_ratio":
                self.roi_padding_ratio = float(new_value)

            elif key == "detector.full_frame_interval":
                self.full_frame_interval = float(new_value)

            # 发布配置变更事件
            if self.events:
                self.events.publish("detector_config_changed", {
                    'key': key,
                    'old_value': old_value,
                    'new_value': new_value,
                    'timestamp': time.time()
                })

        except Exception as e:
            self.logger.error(f"应用检测器配置变更时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    def get_feature_state(self, feature_name):
        """
        获取检测器特定功能的状态

        Args:
            feature_name: 功能名称 (例如 'mediapipe')

        Returns:
            bool: 功能当前状态
        """
        try:
            if feature_name == 'mediapipe':
                state = self.using_mediapipe
            elif feature_name == 'roi':
                state = self.roi_enabled
            elif feature_name == 'downscale':
                state = self.use_downscale
            else:
                self.logger.warning(f"检测器不支持功能: {feature_name}")

                # 发布不支持的功能查询事件
                if self.events:
                    self.events.publish("unsupported_feature_query", {
                        'feature_name': feature_name,
                        'timestamp': time.time()
                    })

                return False

            # 发布功能状态查询事件
            if self.events:
                self.events.publish("feature_state_queried", {
                    'feature_name': feature_name,
                    'state': state,
                    'timestamp': time.time()
                })

            return state
        except Exception as e:
            self.logger.error(f"获取检测器功能状态时出错: {e}")

            # 发布功能查询错误事件
            if self.events:
                self.events.publish("feature_query_error", {
                    'feature_name': feature_name,
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False


    def toggle_feature(self, feature_name, state):
        """
        切换检测器特定功能

        Args:
            feature_name: 功能名称
            state: 要设置的状态 (True/False)

        Returns:
            bool: 是否成功切换功能
        """
        try:
            if feature_name == 'mediapipe':
                # 记录当前状态
                old_state = self.using_mediapipe

                # 设置新状态
                self.using_mediapipe = state

                # 如果启用但未初始化，则初始化
                if state and not hasattr(self, 'pose'):
                    self._init_mediapipe()

                self.logger.info(
                    f"MediaPipe功能已{'启用' if state else '禁用'}")

                # 发布功能切换事件
                if self.events:
                    self.events.publish("feature_toggled", {
                        'feature_name': feature_name,
                        'old_state': old_state,
                        'new_state': state,
                        'timestamp': time.time()
                    })

                    return True
            elif feature_name == 'roi':
                # 记录当前状态
                old_state = self.roi_enabled

                # 设置新状态
                self.roi_enabled = state

                self.logger.info(
                    f"ROI跟踪已{'启用' if state else '禁用'}")

                # 发布功能切换事件
                if self.events:
                    self.events.publish("feature_toggled", {
                        'feature_name': feature_name,
                        'old_state': old_state,
                        'new_state': state,
                        'timestamp': time.time()
                    })

                return True
            elif feature_name == 'downscale':
                # 记录当前状态
                old_state = self.use_downscale

                # 设置新状态
                self.use_downscale = state

                self.logger.info(
                    f"降采样处理已{'启用' if state else '禁用'}")

                # 发布功能切换事件
                if self.events:
                    self.events.publish("feature_toggled", {
                        'feature_name': feature_name,
                        'old_state': old_state,
                        'new_state': state,
                        'timestamp': time.time()
                    })

                return True
            else:
                self.logger.warning(f"检测器不支持功能: {feature_name}")

                # 发布不支持的功能事件
                if self.events:
                    self.events.publish("unsupported_feature_toggle", {
                        'feature_name': feature_name,
                        'timestamp': time.time()
                    })

                return False
        except Exception as e:
            self.logger.error(f"切换检测器功能时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # 发布功能切换错误事件
            if self.events:
                self.events.publish("feature_toggle_error", {
                    'feature_name': feature_name,
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False


    def release_resources(self):
        """释放检测器使用的资源"""
        try:
            # 收集释放前的资源统计信息
            resource_stats = {}

            # 清理缓存
            if hasattr(self, 'detection_cache'):
                resource_stats['cache_size'] = self.detection_cache.info().get(
                    'size', 0) if hasattr(self.detection_cache,
                                          'info') else 'unknown'
                self.detection_cache.clear()
                self.logger.info("检测缓存已清理")

            # 清理其他资源
            if hasattr(self, 'kalman_filters'):
                resource_stats['kalman_filters_count'] = len(self.kalman_filters)
                self.kalman_filters.clear()

            # 释放MediaPipe资源
            if hasattr(self, 'pose'):
                if hasattr(self.pose, 'close'):
                    self.pose.close()
                del self.pose
                resource_stats['mediapipe_released'] = True

            # 释放YOLO模型资源
            if hasattr(self, 'yolo_pose'):
                if hasattr(self.yolo_pose, 'cpu'):
                    self.yolo_pose.cpu()
                del self.yolo_pose
                resource_stats['yolo_released'] = True

            # 强制垃圾回收
            import gc
            gc.collect()

            self.logger.info("检测器资源已释放")

            # 发布资源释放事件
            if self.events:
                self.events.publish("resources_released", {
                    'resource_stats': resource_stats,
                    'timestamp': time.time()
                })

            return True
        except Exception as e:
            self.logger.error(f"释放检测器资源时出错: {e}")
            import traceback
            traceback.print_exc()

            # 发布资源释放错误事件
            if self.events:
                self.events.publish("resource_release_error", {
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False
