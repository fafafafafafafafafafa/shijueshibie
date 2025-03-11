import math
import time
import numpy as np
import cv2
from utils.data_structures import CircularBuffer
from utils.cache_utils import create_standard_cache, generate_cache_key
from interface.position_mapper_interface import PositionMapperInterface


class SimplifiedPositionMapper(PositionMapperInterface):
    """
    综合位置映射器，整合多个功能模块
    使用组合模式代替多继承，提高代码清晰度和性能
    集成事件系统，发布位置映射的关键事件
    """

    def __init__(self, room_width=800, room_height=600, event_system=None):
        """初始化位置映射器"""
        # 保存事件系统
        self.events = event_system

        # 基本配置
        self.room_width = room_width
        self.room_height = room_height

        # 缓存系统 - 使用标准化函数
        self.position_cache = create_standard_cache(
            name="position_mapper",
            capacity=30,
            timeout=0.5,
            persistent=True
        )

        # 校准数据
        self.calibration_height = None
        self.min_height = 50
        self.max_height = 400
        self.depth_scale = 8.0

        # 历史数据和状态
        self.position_history = CircularBuffer(45)
        self.last_stable_position = None
        self.last_valid_position = None
        self.last_update_time = time.time()
        self.motion_state = "static"  # 'static', 'moving', 'occluded'
        self.state_duration = 0  # 当前状态持续时间（帧数）
        self.last_action = "Static"  # 上一个识别的动作

        # 遮挡检测部分
        self._init_occlusion_detector()

        # 卡尔曼滤波部分
        self._init_kalman_tracker()

        # 运动预测部分
        self._init_motion_predictor()

        # 位置平滑部分
        self._init_position_smoother()

        # 后退检测
        self.backward_exponent = 2.2
        self.normal_exponent = 1.8
        self.backward_enabled = True
        self.backward_counter = 0

        # 调试信息
        self.debug_info = {
            "occlusion": False,
            "motion": "static",
            "vel": (0, 0),
            "acc": (0, 0),
            "kalman": False,
            "smooth": 5,
            "occlusion_confidence": 0.0,
            "predicted": False
        }

        # 设置一个标志，用于调试和故障排除
        self.debug_mode = False

        print("Integrated PositionMapper initialized")

        # 发布初始化完成事件
        if self.events:
            self.events.publish("position_mapper_initialized", {
                'room_width': room_width,
                'room_height': room_height,
                'timestamp': time.time()
            })

    def _init_occlusion_detector(self):
        """初始化遮挡检测器部分"""
        # 遮挡状态参数
        self.occlusion_detected = False
        self.occlusion_counter = 0
        self.max_occlusion_frames = 30  # 增加最大遮挡帧数
        self.occlusion_confidence = 0.0
        self.recovery_counter = 0  # 恢复过程计数器
        self.max_recovery_frames = 10  # 恢复过程最大帧数

        # 发布遮挡检测器初始化事件
        if self.events:
            self.events.publish("occlusion_detector_initialized", {
                'max_occlusion_frames': self.max_occlusion_frames,
                'max_recovery_frames': self.max_recovery_frames,
                'timestamp': time.time()
            })

    def _init_kalman_tracker(self):
        """初始化卡尔曼跟踪器部分"""
        # 初始化卡尔曼滤波器
        self.kalman_filter = None
        self.initialized = False
        success = self._create_kalman_filter()

        # 发布卡尔曼跟踪器初始化事件
        if self.events:
            self.events.publish("kalman_tracker_initialized", {
                'success': success,
                'timestamp': time.time()
            })

    def _init_motion_predictor(self):
        """初始化运动预测器部分"""
        self.velocity = (0, 0)  # (vx, vy)
        self.acceleration = (0, 0)  # (ax, ay)
        self.velocity_history = CircularBuffer(10)
        self.motion_last_update_time = 0

        # 发布运动预测器初始化事件
        if self.events:
            self.events.publish("motion_predictor_initialized", {
                'velocity_history_capacity': 10,
                'timestamp': time.time()
            })

    def _init_position_smoother(self):
        """初始化位置平滑器部分"""
        # 位置平滑器不需要特殊初始化
        # 平滑参数将在_determine_smooth_factor和平滑方法中设置
        self.use_smoothing = True  # 默认启用平滑

        # 发布位置平滑器初始化事件
        if self.events:
            self.events.publish("position_smoother_initialized", {
                'timestamp': time.time()
            })

    def _create_kalman_filter(self):
        """创建卡尔曼滤波器 - 使用恒加速度模型"""
        try:
            # 状态向量: [x, y, vx, vy, ax, ay] - 位置、速度和加速度
            self.kalman_filter = cv2.KalmanFilter(6, 2)

            # 测量矩阵 - 只测量位置 (x,y)
            self.kalman_filter.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ], np.float32)

            # 转移矩阵 - 恒加速度模型
            self.kalman_filter.transitionMatrix = np.array([
                [1, 0, 1, 0, 0.5, 0],  # x = x + vx + 0.5*ax
                [0, 1, 0, 1, 0, 0.5],  # y = y + vy + 0.5*ay
                [0, 0, 1, 0, 1, 0],  # vx = vx + ax
                [0, 0, 0, 1, 0, 1],  # vy = vy + ay
                [0, 0, 0, 0, 1, 0],  # ax = ax
                [0, 0, 0, 0, 0, 1]  # ay = ay
            ], np.float32)

            # 过程噪声协方差 - 初始值适中
            self.kalman_filter.processNoiseCov = np.array([
                [0.01, 0, 0, 0, 0, 0],
                [0, 0.01, 0, 0, 0, 0],
                [0, 0, 0.01, 0, 0, 0],
                [0, 0, 0, 0.01, 0, 0],
                [0, 0, 0, 0, 0.01, 0],
                [0, 0, 0, 0, 0, 0.01]
            ], np.float32)

            # 测量噪声协方差
            self.kalman_filter.measurementNoiseCov = np.array([
                [0.1, 0],
                [0, 0.1]
            ], np.float32)

            # 后验错误协方差
            self.kalman_filter.errorCovPost = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],np.float32)

            # 发布卡尔曼滤波器创建成功事件
            if self.events:
                self.events.publish("kalman_filter_created", {
                    'timestamp': time.time()
                })

            return True
        except Exception as e:
            print(f"Failed to initialize Kalman filter: {e}")
            self.kalman_filter = None

            # 发布卡尔曼滤波器创建失败事件
            if self.events:
                self.events.publish("kalman_filter_creation_failed", {
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False

    def initialize_tracker(self, x, y):
        """使用初始位置初始化跟踪器"""
        if self.kalman_filter is None:
            if not self._create_kalman_filter():
                # 发布初始化失败事件
                if self.events:
                    self.events.publish("tracker_initialization_failed", {
                        'reason': 'no_kalman_filter',
                        'timestamp': time.time()
                    })
                return False

        try:
            # 初始化状态向量
            self.kalman_filter.statePost = np.array([
                [float(x)],
                [float(y)],
                [0],  # 初始vx
                [0],  # 初始vy
                [0],  # 初始ax
                [0]  # 初始ay
            ], np.float32)

            self.initialized = True

            # 发布初始化成功事件
            if self.events:
                self.events.publish("tracker_initialized", {
                    'position': (x, y),
                    'timestamp': time.time()
                })

            return True
        except Exception as e:
            print(f"Failed to initialize Kalman tracker: {e}")

            # 发布初始化失败事件
            if self.events:
                self.events.publish("tracker_initialization_failed", {
                    'error': str(e),
                    'position': (x, y),
                    'timestamp': time.time()
                })

            return False

    def detect_occlusion(self, person, position_history, last_valid_position):
        """
        检测遮挡状态

        Args:
            person: 人体检测数据字典
            position_history: 历史位置记录
            last_valid_position: 上一个有效位置

        Returns:
            tuple: (is_occluded, confidence) 遮挡状态和置信度
        """
        # 初始化置信度
        confidence = 0.0
        confidence_factors = []

        # 如果显式标记为历史数据，认为存在遮挡
        if person.get('is_history', False):
            # 发布历史数据遮挡事件
            if self.events:
                self.events.publish("historical_data_occlusion", {
                    'confidence': 0.8,
                    'timestamp': time.time()
                })
            return True, 0.8

        # 1. 检查边界框突然变化
        if 'bbox' in person and 'last_bbox' in person:
            x1, y1, x2, y2 = person['bbox']
            lx1, ly1, lx2, ly2 = person.get('last_bbox', (0, 0, 0, 0))

            # 计算当前和上一帧的面积
            current_area = (x2 - x1) * (y2 - y1)
            last_area = (lx2 - lx1) * (ly2 - ly1)

            # 检测面积急剧减少
            if last_area > 0 and current_area < last_area * 0.8:
                area_factor = min(1.0, 1.0 - (current_area / last_area))
                confidence_factors.append(('area', area_factor * 0.8))

            # 检测位置突变
            current_center_x = (x1 + x2) / 2
            current_center_y = (y1 + y2) / 2
            last_center_x = (lx1 + lx2) / 2
            last_center_y = (ly1 + ly2) / 2

            center_shift = math.sqrt((current_center_x - last_center_x) ** 2 +
                                     (current_center_y - last_center_y) ** 2)

            # 如果中心点移动过大，可能是检测错误或部分遮挡
            if center_shift > 50:  # 阈值可调
                shift_factor = min(1.0, center_shift / 200)  # 归一化到0-1
                confidence_factors.append(('shift', shift_factor * 0.7))

        # 2. 检查关键点完整性下降
        if 'keypoints' in person:
            # 计算有效关键点数量
            valid_keypoints = sum(
                1 for _, _, conf in person['keypoints'] if conf > 0.5)
            expected_keypoints = 12  # 一般至少应该有这么多关键点

            if valid_keypoints < expected_keypoints * 0.8:
                kp_factor = 0.3 + 0.7 * (
                        1 - valid_keypoints / expected_keypoints)
                confidence_factors.append(('keypoints', kp_factor))

        # 3. 检测前几帧的稳定性突然下降
        if len(position_history) >= 3 and last_valid_position:
            recent_positions = list(position_history)[-3:]

            # 计算最近几帧位置变化的方差
            diffs = []
            for i in range(len(recent_positions) - 1):
                dx = recent_positions[i + 1][0] - recent_positions[i][0]
                dy = recent_positions[i + 1][1] - recent_positions[i][1]
                diffs.append(math.sqrt(dx * dx + dy * dy))

            if diffs and max(diffs) > 30:  # 如果有突然的大幅跳变
                stability_factor = min(1.0, max(diffs) / 100)
                confidence_factors.append(('stability', stability_factor * 0.5))

        # 4. 考虑遮挡历史
        if self.occlusion_detected:
            # 如果已经检测到遮挡，增加新一帧也是遮挡的可能性
            history_factor = max(0.0, 0.3 - 0.05 * (
                    self.occlusion_counter / self.max_occlusion_frames))
            confidence_factors.append(('history', history_factor))

        # 综合多种因素，计算最终置信度
        if confidence_factors:
            # 使用加权平均
            total_weight = 0
            weighted_sum = 0

            for factor_name, factor_value in confidence_factors:
                # 可以给不同因素分配不同权重
                weight = 1.0
                if factor_name == 'keypoints':
                    weight = 1.5  # 关键点因素权重更高
                elif factor_name == 'area':
                    weight = 1.2  # 面积变化也很重要

                weighted_sum += factor_value * weight
                total_weight += weight

            confidence = weighted_sum / total_weight

        # 应用阈值判断 - 增加阈值，降低误检测
        is_occluded = confidence > 0.45  # 增加阈值，减少误触发

        # 状态变化时发布事件
        if self.events:
            if is_occluded and not self.occlusion_detected:
                self.events.publish("occlusion_started", {
                    'confidence': confidence,
                    'factors': confidence_factors,
                    'timestamp': time.time()
                })
            elif not is_occluded and self.occlusion_detected:
                self.events.publish("occlusion_ended", {
                    'duration': self.occlusion_counter / 30.0,  # 假设30fps
                    'timestamp': time.time()
                })

        return is_occluded, confidence

    def update_occlusion_state(self, is_occluded, confidence, person):
        """
        根据检测结果更新遮挡状态，使用平滑的状态转换

        Args:
            is_occluded: 是否检测到遮挡
            confidence: 遮挡置信度
            person: 人体检测数据字典
        """
        previous_state = {
            'occlusion_detected': self.occlusion_detected,
            'occlusion_counter': self.occlusion_counter,
            'occlusion_confidence': self.occlusion_confidence,
            'recovery_counter': self.recovery_counter
        }

        if is_occluded:
            if not self.occlusion_detected:
                # 新遮挡开始
                self.occlusion_detected = True
                self.occlusion_counter = 1
                self.occlusion_confidence = confidence
                self.recovery_counter = 0
                person['occlusion_start'] = True
                if self.debug_mode:
                    print(
                        f"Occlusion started with confidence: {confidence:.2f}")

                # 发布遮挡开始事件
                if self.events:
                    self.events.publish("new_occlusion_started", {
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
            else:
                # 遮挡持续中
                self.occlusion_counter = min(self.occlusion_counter + 1,
                                             self.max_occlusion_frames)
                # 平滑更新置信度
                self.occlusion_confidence = confidence * 0.3 + self.occlusion_confidence * 0.7
                person['occlusion_continued'] = True

                # 发布遮挡持续事件（每5帧一次，以减少事件数量）
                if self.events and self.occlusion_counter % 5 == 0:
                    self.events.publish("occlusion_continuing", {
                        'counter': self.occlusion_counter,
                        'confidence': self.occlusion_confidence,
                        'max_frames': self.max_occlusion_frames,
                        'timestamp': time.time()
                    })
        else:
            # 没有检测到遮挡
            if self.occlusion_detected:
                # 开始恢复过程
                self.recovery_counter += 1

                # 降低遮挡置信度
                self.occlusion_confidence *= 0.8

                # 判断是否完全恢复
                if self.recovery_counter >= self.max_recovery_frames or self.occlusion_confidence < 0.2:
                    # 完成恢复
                    self.occlusion_detected = False
                    self.occlusion_confidence = 0.0
                    self.occlusion_counter = 0
                    self.recovery_counter = 0
                    person['occlusion_ended'] = True
                    if self.debug_mode:
                        print("Occlusion ended")

                    # 发布遮挡结束事件
                    if self.events:
                        self.events.publish("occlusion_recovery_complete", {
                            'recovery_frames': self.recovery_counter,
                            'timestamp': time.time()
                        })
                else:
                    # 恢复过程中
                    person['occlusion_recovering'] = True
                    # 随着恢复的进行，减少遮挡计数
                    self.occlusion_counter = max(0, self.occlusion_counter - 1)

                    # 发布恢复进行中事件
                    if self.events:
                        self.events.publish("occlusion_recovering", {
                            'recovery_counter': self.recovery_counter,
                            'max_recovery_frames': self.max_recovery_frames,
                            'occlusion_confidence': self.occlusion_confidence,
                            'timestamp': time.time()
                        })
            else:
                # 持续无遮挡状态
                self.recovery_counter = 0

        # 发布状态变化事件
        if self.events and (
                previous_state[
                    'occlusion_detected'] != self.occlusion_detected or
                abs(previous_state[
                        'occlusion_confidence'] - self.occlusion_confidence) > 0.1 or
                previous_state['recovery_counter'] != self.recovery_counter
        ):
            self.events.publish("occlusion_state_updated", {
                'previous_state': previous_state,
                'current_state': {
                    'occlusion_detected': self.occlusion_detected,
                    'occlusion_counter': self.occlusion_counter,
                    'occlusion_confidence': self.occlusion_confidence,
                    'recovery_counter': self.recovery_counter
                },
                'timestamp': time.time()
            })

    def is_occluded(self):
        """获取当前是否处于遮挡状态"""
        return self.occlusion_detected

    def get_occlusion_counter(self):
        """获取遮挡持续的帧数"""
        return self.occlusion_counter

    def get_max_occlusion_frames(self):
        """获取最大遮挡帧数"""
        return self.max_occlusion_frames

    def get_occlusion_confidence(self):
        """获取当前遮挡置信度"""
        return self.occlusion_confidence

    def set_calibration(self, height):
        """设置校准高度"""
        old_height = self.calibration_height

        if height <= 0:
            print("Warning: Invalid calibration height. Using default.")
            self.calibration_height = 200
        else:
            self.calibration_height = height
            print(f"Calibration height set to: {height}")

        # 发布校准设置事件
        if self.events:
            self.events.publish("calibration_height_set", {
                'old_height': old_height,
                'new_height': self.calibration_height,
                'is_valid': height > 0,
                'timestamp': time.time()
            })

    def map_position_to_room(self, frame_width, frame_height, room_width,
                             room_height, person):
        """将摄像头坐标映射到房间平面图坐标"""
        mapping_start_time = time.time()

        try:
            # 创建缓存键
            cache_key = None
            if all(k in person for k in ['center_x', 'center_y', 'height']):
                # 使用新的键生成函数
                cache_data = {
                    'center_x': person['center_x'],
                    'center_y': person['center_y'],
                    'height': person['height'],
                    'frame_dim': (frame_width, frame_height),
                    'room_dim': (room_width, room_height)
                }
                cache_key = generate_cache_key(cache_data, prefix="position")

                # 检查缓存
                cached_position = self.position_cache.get(cache_key)
                if cached_position is not None:
                    # 发布缓存命中事件
                    if self.events:
                        self.events.publish("position_cache_hit", {
                            'position': cached_position,
                            'timestamp': time.time()
                        })

                    return cached_position

            # 发布缓存未命中事件
            if self.events and cache_key:
                self.events.publish("position_cache_miss", {
                    'timestamp': time.time()
                })

            # 提取关键数据
            center_x = person['center_x']
            center_y = person['center_y']
            person_height = person['height']
            current_time = person.get('detection_time', time.time())

            # 简单直接的位置计算（用于故障排除）
            if getattr(self, 'simple_mapping_mode', False):
                room_x = int(center_x / frame_width * room_width)
                room_y = int(center_y / frame_height * room_height)

                # 发布简单映射事件
                if self.events:
                    self.events.publish("simple_mapping_used", {
                        'input': (center_x, center_y),
                        'output': (room_x, room_y),
                        'timestamp': time.time()
                    })

                return room_x, room_y, 1.0

            # 更新人体数据历史
            if 'bbox' in person:
                person['last_bbox'] = person.get('last_bbox', person['bbox'])

            # 检测遮挡状态
            is_occluded, confidence = self.detect_occlusion(
                person, self.position_history, self.last_valid_position)
            self.update_occlusion_state(is_occluded, confidence, person)
            self.debug_info["occlusion"] = self.occlusion_detected
            self.debug_info["occlusion_confidence"] = round(
                self.occlusion_confidence, 2)

            # 更新动作状态
            action = person.get('action', self.last_action)
            self.update_motion_state(action, current_time)

            # 检测是否后退
            is_backward = self._detect_backward_motion()

            # 发布后退检测事件
            if self.events and is_backward:
                self.events.publish("backward_motion_detected", {
                    'backward_counter': self.backward_counter,
                    'timestamp': time.time()
                })

            # 计算深度值
            if self.calibration_height and self.calibration_height > 0:
                relative_height = person_height / self.calibration_height
                depth = (1.0 / relative_height) * self.depth_scale + 2.0
            else:
                # 归一化高度计算深度
                normalized_height = (person_height - self.min_height) / (
                        self.max_height - self.min_height)
                normalized_height = max(0.1, min(normalized_height, 1.0))
                depth = (1.0 / normalized_height) * self.depth_scale + 2.0

            # 限制深度值范围
            depth = max(0.1, min(depth, self.depth_scale * 2))
            person['depth'] = depth  # 保存深度信息供后续使用

            # 水平坐标映射
            room_x = int(center_x / frame_width * room_width)

            # 垂直坐标映射 - 选择适当的非线性指数
            min_y = 20
            max_y = room_height - 20
            normalized_depth = depth / (self.depth_scale * 2)

            # 根据是否检测到后退选择指数
            exponent = self.backward_exponent if is_backward else self.normal_exponent
            room_y = int(
                min_y + (max_y - min_y) * (normalized_depth ** exponent))

            # 发布深度计算事件
            if self.events:
                self.events.publish("depth_calculated", {
                    'person_height': person_height,
                    'calibration_height': self.calibration_height,
                    'normalized_depth': normalized_depth,
                    'depth': depth,
                    'exponent': exponent,
                    'is_backward': is_backward,
                    'timestamp': time.time()
                })

            # 处理遮挡情况下的位置
            if self.occlusion_detected and self.last_stable_position:
                # 使用运动预测器计算预测位置
                elapsed_time = 0.033  # 假设30fps
                predicted_position = self.predict_position(elapsed_time)

                if predicted_position:
                    # 使用S形混合因子而不是线性混合
                    occlusion_progress = self.occlusion_counter / self.max_occlusion_frames
                    sigmoid_value = 1.0 / (
                            1.0 + math.exp(-6 * (occlusion_progress - 0.5)))
                    blend_factor = 0.3 + 0.6 * sigmoid_value
                    blend_factor = min(0.9, blend_factor)

                    # 混合实际位置和预测位置
                    room_x = int(
                        room_x * (1 - blend_factor) + predicted_position[
                            0] * blend_factor)
                    room_y = int(
                        room_y * (1 - blend_factor) + predicted_position[
                            1] * blend_factor)

                    # 记录使用了预测位置
                    person['using_predicted'] = True
                    person['blend_factor'] = blend_factor
                    self.debug_info["predicted"] = True

                    # 发布使用预测位置事件
                    if self.events:
                        self.events.publish("predicted_position_used", {
                            'original_position': (room_x, room_y),
                            'predicted_position': predicted_position,
                            'blend_factor': blend_factor,
                            'occlusion_progress': occlusion_progress,
                            'final_position': (room_x, room_y),
                            'timestamp': time.time()
                        })
                else:
                    self.debug_info["predicted"] = False
            else:
                self.debug_info["predicted"] = False

            # 确保位置在房间范围内
            room_x = max(0, min(room_x, room_width - 1))
            room_y = max(0, min(room_y, room_height - 1))

            # 估计速度和加速度
            self.update_motion(
                (room_x, room_y),
                current_time,
                self.last_stable_position,
                self.motion_state,
                self.occlusion_counter,
                self.max_occlusion_frames
            )

            self.debug_info["vel"] = (
                round(self.velocity[0], 1), round(self.velocity[1], 1))
            self.debug_info["acc"] = (
                round(self.acceleration[0], 1), round(self.acceleration[1], 1))

            # 检测位置跳变 (用于调试)
            if self.last_valid_position:
                dx = room_x - self.last_valid_position[0]
                dy = room_y - self.last_valid_position[1]
                jump_distance_sq = dx * dx + dy * dy
                if jump_distance_sq > 10000:  # 100^2 像素
                    jump_distance = math.sqrt(jump_distance_sq)
                    if self.debug_mode:
                        print(
                            f"Warning: Position jump detected ({jump_distance:.1f} pixels)")

                    # 发布位置跳变事件
                    if self.events:
                        self.events.publish("position_jump_detected", {
                            'previous_position': self.last_valid_position,
                            'current_position': (room_x, room_y),
                            'jump_distance': jump_distance,
                            'timestamp': time.time()
                        })

            # 应用平滑
            smooth_factor = self._determine_smooth_factor()
            self.debug_info["smooth"] = smooth_factor

            # 根据运动状态选择平滑方法
            if self.motion_state == "occluded":
                smoothed_x, smoothed_y = self._gaussian_smooth(room_x, room_y,
                                                               smooth_factor)
            elif self.motion_state == "static":
                smoothed_x, smoothed_y = self._static_smooth(room_x, room_y,
                                                             smooth_factor)
            elif self.motion_state == "moving":
                smoothed_x, smoothed_y = self._moving_smooth(room_x, room_y,
                                                             smooth_factor)
            else:
                smoothed_x, smoothed_y = self._basic_smooth(room_x, room_y,
                                                            smooth_factor)

            # 更新位置历史
            self.position_history.append((smoothed_x, smoothed_y))
            self.last_valid_position = (smoothed_x, smoothed_y)

            if self.debug_mode:
                print(
                    f"Mapped position: ({smoothed_x}, {smoothed_y}), depth: {depth:.2f}")

            # 发布位置映射完成事件
            if self.events:
                self.events.publish("position_mapped", {
                    'person': person,
                    'original_position': (center_x, center_y),
                    'room_position': (room_x, room_y),
                    'smoothed_position': (smoothed_x, smoothed_y),
                    'depth': depth,
                    'motion_state': self.motion_state,
                    'occlusion_detected': self.occlusion_detected,
                    'processing_time': time.time() - mapping_start_time,
                    'timestamp': time.time()
                })

            # 缓存结果
            result = (smoothed_x, smoothed_y, depth)
            if cache_key is not None:
                self.position_cache.put(cache_key, result)

            return result

        except Exception as e:
            print(f"Error in position mapping: {e}")
            import traceback
            traceback.print_exc()

            # 发布映射错误事件
            if self.events:
                self.events.publish("position_mapping_error", {
                    'error': str(e),
                    'person_data': {
                        'center': (person.get('center_x', 0), person.get('center_y', 0)),
                        'height': person.get('height', 0)
                    } if 'center_x' in person and 'center_y' in person else {},
                    'timestamp': time.time()
                })

            # 故障安全返回 - 提供一个有效的默认位置
            default_x = room_width // 2
            default_y = room_height // 2

            if self.last_valid_position:
                # 如果有上一个有效位置，使用它
                return self.last_valid_position[0], self.last_valid_position[1], 5.0
            else:
                # 返回房间中心
                return default_x, default_y, 5.0


    def get_stable_position(self, x, y, action=None):
        """
        获取稳定的位置（减少抖动）

        Args:
            x: x坐标
            y: y坐标
            action: 可选的动作类型字符串

        Returns:
            tuple: 稳定后的(x, y)坐标
        """
        # 更新动作状态
        self.last_action = action if action else self.last_action
        self.update_motion_state(self.last_action, time.time())

        # 初始化位置
        if self.last_stable_position is None:
            self.last_stable_position = (x, y)
            self.initialize_tracker(x, y)

            # 发布首次稳定位置事件
            if self.events:
                self.events.publish("initial_stable_position_set", {
                    'position': (x, y),
                    'timestamp': time.time()
                })

            return x, y

        # 使用卡尔曼跟踪器获取稳定位置
        kalman_result = self.update_tracking(
            x, y,
            self.motion_state,
            self.occlusion_counter,
            self.max_occlusion_frames
        )

        if kalman_result:
            # 卡尔曼滤波成功
            kf_x, kf_y = kalman_result
            self.last_stable_position = (kf_x, kf_y)
            self.debug_info["kalman"] = True

            # 发布卡尔曼滤波器稳定位置事件
            if self.events:
                self.events.publish("kalman_stable_position", {
                    'input_position': (x, y),
                    'stable_position': (kf_x, kf_y),
                    'motion_state': self.motion_state,
                    'timestamp': time.time()
                })

            return kf_x, kf_y
        else:
            # 卡尔曼滤波失败，使用简单平滑
            last_x, last_y = self.last_stable_position
            dx = x - last_x
            dy = y - last_y

            # 动态调整平滑因子
            if self.motion_state == "moving":
                smooth_factor = 0.5  # 更快响应
            elif self.motion_state == "occluded":
                smooth_factor = 0.3  # 中度响应
            else:  # static
                smooth_factor = 0.2  # 更稳定

            # 计算新位置
            new_x = last_x + int(dx * smooth_factor)
            new_y = last_y + int(dy * smooth_factor)

            # 确保位置在房间范围内
            new_x = max(0, min(new_x, self.room_width - 1))
            new_y = max(0, min(new_y, self.room_height - 1))

            self.last_stable_position = (new_x, new_y)
            self.debug_info["kalman"] = False

            # 发布简单平滑稳定位置事件
            if self.events:
                self.events.publish("simple_smooth_stable_position", {
                    'input_position': (x, y),
                    'last_position': (last_x, last_y),
                    'stable_position': (new_x, new_y),
                    'smooth_factor': smooth_factor,
                    'motion_state': self.motion_state,
                    'timestamp': time.time()
                })

            return new_x, new_y

    def _determine_smooth_factor(self):
        """根据当前状态动态确定平滑因子"""
        # 如果禁用了平滑，返回最小平滑因子
        if hasattr(self, 'use_smoothing') and not self.use_smoothing:
            return 2

        # 根据运动状态动态调整平滑因子
        if self.motion_state == "moving":
            return 5  # 移动时使用较小的平滑窗口
        elif self.motion_state == "occluded":
            # 遮挡时使用更大的平滑窗口，随着遮挡时间增加而增大
            occlusion_progress = min(1.0,
                                     self.occlusion_counter / self.max_occlusion_frames)
            return 7 + int(5 * occlusion_progress)  # 7-12范围
        else:  # static
            return 6  # 静止时使用中等平滑窗口

    def smooth_position(self, x, y, motion_state=None, occlusion_counter=None,
                        max_occlusion_frames=None):
        """
        外部接口：平滑位置数据

        Args:
            x: X坐标
            y: Y坐标
            motion_state: 运动状态，如果为None则使用当前状态
            occlusion_counter: 遮挡计数，如果为None则使用当前计数
            max_occlusion_frames: 最大遮挡帧数，如果为None则使用当前设置

        Returns:
            tuple: 平滑后的位置坐标 (x, y)
        """
        # 检查是否禁用平滑
        if hasattr(self, 'use_smoothing') and not self.use_smoothing:
            return x, y

        # 使用传入的参数或当前状态
        state = motion_state if motion_state is not None else self.motion_state
        occ_counter = occlusion_counter if occlusion_counter is not None else self.occlusion_counter
        max_frames = max_occlusion_frames if max_occlusion_frames is not None else self.max_occlusion_frames

        # 获取平滑因子
        smooth_factor = self._determine_smooth_factor()
        self.debug_info["smooth"] = smooth_factor

        # 根据状态选择平滑方法
        if state == "occluded":
            return self._gaussian_smooth(x, y, smooth_factor)
        elif state == "static":
            return self._static_smooth(x, y, smooth_factor)
        elif state == "moving":
            return self._moving_smooth(x, y, smooth_factor)
        else:
            return self._basic_smooth(x, y, smooth_factor)

    def _gaussian_smooth(self, x, y, smooth_factor):
        """高斯加权平滑，适用于遮挡状态"""
        if len(self.position_history) < 3:
            return x, y

        # 获取最近的历史位置
        positions = self.position_history.get_latest(
            min(len(self.position_history), smooth_factor))

        if len(positions) < 3:
            return self._basic_smooth(x, y, smooth_factor)

        # 遮挡情况下使用高斯加权
        occlusion_progress = min(1.0,
                                 self.occlusion_counter / self.max_occlusion_frames)

        # 中心点设置在历史中间偏后位置
        center_ratio = 0.6 - 0.2 * occlusion_progress  # 从0.6降到0.4
        center_idx = int(len(positions) * center_ratio)

        # 控制高斯分布宽度
        sigma = len(positions) / (3.0 - occlusion_progress)

        # 计算高斯权重
        total_x, total_y, total_weight = 0, 0, 0
        for i, (px, py) in enumerate(positions):
            # 计算高斯权重
            weight = math.exp(-0.5 * ((i - center_idx) / sigma) ** 2)
            total_x += px * weight
            total_y += py * weight
            total_weight += weight

        if total_weight > 0:
            gaussian_x = total_x / total_weight
            gaussian_y = total_y / total_weight

            # 混合当前位置和历史加权平均
            blend = max(0.2, 0.6 - 0.4 * occlusion_progress)  # 从0.6降到0.2
            smooth_x = int(x * blend + gaussian_x * (1 - blend))
            smooth_y = int(y * blend + gaussian_y * (1 - blend))
            return smooth_x, smooth_y

        return x, y

    def _static_smooth(self, x, y, smooth_factor):
        """静止状态平滑 - 使用更强的平滑，抑制微小抖动"""
        # 获取最近的历史位置
        positions = self.position_history.get_latest(
            min(len(self.position_history), smooth_factor + 2))

        # 计算历史位置的方差，判断稳定性
        if len(positions) >= 3:
            last_pos = positions[-1]

            # 计算坐标方差
            var_x = sum((p[0] - last_pos[0]) ** 2 for p in positions) / len(
                positions)
            var_y = sum((p[1] - last_pos[1]) ** 2 for p in positions) / len(
                positions)
            variance = var_x + var_y

            # 如果方差很小，说明位置非常稳定，强制固定在最后的稳定位置
            if variance < 10:  # 阈值可调
                return last_pos

        # 使用简单平均平滑
        if positions:
            total_x = sum(p[0] for p in positions)
            total_y = sum(p[1] for p in positions)
            avg_x = total_x / len(positions)
            avg_y = total_y / len(positions)

            # 混合当前位置(30%)和历史平均(70%)
            blend = 0.3  # 静止状态下更倾向于历史平均
            smooth_x = int(x * blend + avg_x * (1 - blend))
            smooth_y = int(y * blend + avg_y * (1 - blend))
            return smooth_x, smooth_y

        return x, y

    def _moving_smooth(self, x, y, smooth_factor):
        """移动状态平滑 - 使用较轻的平滑，保持响应性"""
        # 使用较少的历史点
        reduced_factor = max(3, smooth_factor - 2)
        positions = self.position_history.get_latest(
            min(len(self.position_history), reduced_factor))

        if not positions:
            return x, y

        # 创建平方增长权重 - 最新的位置权重显著更高
        weights = [(i + 1) ** 2 for i in range(len(positions))]
        total_weight = sum(weights)

        # 使用加权平均
        weighted_x = sum(p[0] * w for p, w in zip(positions, weights))
        weighted_y = sum(p[1] * w for p, w in zip(positions, weights))

        if total_weight > 0:
            avg_x = weighted_x / total_weight
            avg_y = weighted_y / total_weight

            # 混合当前位置(70%)和历史平均(30%)
            blend = 0.7
            smooth_x = int(x * blend + avg_x * (1 - blend))
            smooth_y = int(y * blend + avg_y * (1 - blend))
            return smooth_x, smooth_y

        return x, y

    def _basic_smooth(self, x, y, smooth_factor):
        """基本平滑方法 - 线性加权平均"""
        # 获取最近的历史位置
        positions = self.position_history.get_latest(
            min(len(self.position_history), smooth_factor))

        if not positions:
            return x, y

        # 线性加权 - 越近的点权重越大
        total_x, total_y, total_weight = 0, 0, 0
        for i, (px, py) in enumerate(positions):
            weight = i + 1  # 线性递增权重
            total_x += px * weight
            total_y += py * weight
            total_weight += weight

        if total_weight > 0:
            avg_x = total_x / total_weight
            avg_y = total_y / total_weight

            # 混合当前位置和历史平均
            blend = 0.5  # 50%当前值，50%历史平均
            smooth_x = int(x * blend + avg_x * (1 - blend))
            smooth_y = int(y * blend + avg_y * (1 - blend))
            return smooth_x, smooth_y

        return x, y

    def update_motion(self, current_position, current_time, last_position,
                      motion_state, occlusion_counter=0, max_occlusion_frames=30):
        """更新运动状态，计算速度和加速度"""
        # 确保有历史位置
        if last_position is None:
            self.motion_last_update_time = current_time
            return (0, 0), (0, 0)

        # 计算时间间隔
        dt = current_time - self.motion_last_update_time
        if dt <= 0:
            dt = 0.033  # 假设30fps

        # 计算当前速度
        dx = current_position[0] - last_position[0]
        dy = current_position[1] - last_position[1]

        # 计算速度 (像素/秒)
        current_vx = dx / dt
        current_vy = dy / dt

        # 计算加速度 (像素/秒²)
        if self.velocity != (0, 0):
            current_ax = (current_vx - self.velocity[0]) / dt
            current_ay = (current_vy - self.velocity[1]) / dt
        else:
            current_ax, current_ay = 0, 0

        # 根据运动状态选择平滑因子
        if motion_state == "moving":
            vel_alpha = 0.7  # 移动时快速响应速度变化
            acc_alpha = 0.5  # 移动时中等响应加速度变化
        elif motion_state == "occluded":
            # 遮挡时使用动态平滑因子
            occlusion_progress = min(1.0,
                                     occlusion_counter / max_occlusion_frames)
            vel_alpha = max(0.2, 0.6 - 0.4 * occlusion_progress)  # 从0.6降到0.2
            acc_alpha = max(0.1, 0.4 - 0.3 * occlusion_progress)  # 从0.4降到0.1
        else:  # static
            vel_alpha = 0.3  # 静止时中等响应速度变化
            acc_alpha = 0.2  # 静止时较慢响应加速度变化

        # 平滑速度和加速度
        smooth_vx = current_vx * vel_alpha + self.velocity[0] * (1 - vel_alpha)
        smooth_vy = current_vy * vel_alpha + self.velocity[1] * (1 - vel_alpha)

        smooth_ax = current_ax * acc_alpha + self.acceleration[0] * (
                1 - acc_alpha)
        smooth_ay = current_ay * acc_alpha + self.acceleration[1] * (
                1 - acc_alpha)

        # 使用多帧平均值进一步平滑速度
        if len(self.velocity_history) >= 3:
            velocities = list(self.velocity_history)

            # 添加当前速度
            velocities.append((smooth_vx, smooth_vy))

            # 计算加权平均，最近的速度权重更高
            total_vx, total_vy, total_weight = 0, 0, 0
            for i, (vx, vy) in enumerate(velocities):
                weight = i + 1  # 线性递增权重
                total_vx += vx * weight
                total_vy += vy * weight
                total_weight += weight

            if total_weight > 0:
                smooth_vx = total_vx / total_weight
                smooth_vy = total_vy / total_weight

        # 更新速度和加速度
        self.velocity = (smooth_vx, smooth_vy)
        self.acceleration = (smooth_ax, smooth_ay)

        # 更新速度历史
        self.velocity_history.append(self.velocity)

        # 更新时间戳
        self.motion_last_update_time = current_time

        # 发布运动更新事件
        if self.events:
            self.events.publish("motion_updated", {
                'position': current_position,
                'velocity': self.velocity,
                'acceleration': self.acceleration,
                'motion_state': motion_state,
                'timestamp': current_time
            })

        return self.velocity, self.acceleration

    def predict_position(self, elapsed_time):
        """预测未来位置，用于遮挡恢复"""
        if self.last_stable_position is None:
            return None

        # 使用物理公式预测位置
        last_x, last_y = self.last_stable_position
        vx, vy = self.velocity
        ax, ay = self.acceleration

        # 加权平均原始测量和预测值
        if self.is_occluded():
            # 动态调整混合比例
            occlusion_progress = min(1.0,
                                     self.occlusion_counter / self.max_occlusion_frames)
            bez_weight = min(0.8, 0.4 + 0.4 * occlusion_progress)
        else:
            bez_weight = 0.4  # 默认贝塞尔曲线权重

        # 计算物理预测
        pred_x = last_x + vx * elapsed_time + 0.5 * ax * elapsed_time * elapsed_time
        pred_y = last_y + vy * elapsed_time + 0.5 * ay * elapsed_time * elapsed_time

        # 获取最近的位置点用于贝塞尔曲线
        recent_positions = self.position_history.get_latest(4)

        # 如果有足够的历史位置点，使用贝塞尔曲线进行平滑预测
        if len(recent_positions) >= 3:
            # 使用最后3个点创建贝塞尔曲线控制点
            p0 = recent_positions[0]
            p1 = recent_positions[1]
            p2 = recent_positions[2]

            t = min(1.0, elapsed_time)
            mt = 1 - t

            bez_x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
            bez_y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]

            # 混合物理预测和贝塞尔曲线预测
            pred_x = int(pred_x * (1 - bez_weight) + bez_x * bez_weight)
            pred_y = int(pred_y * (1 - bez_weight) + bez_y * bez_weight)

        # 确保位置在房间范围内
        pred_x = max(0, min(pred_x, self.room_width - 1))
        pred_y = max(0, min(pred_y, self.room_height - 1))

        # 发布位置预测事件
        if self.events:
            self.events.publish("position_predicted", {
                'last_position': self.last_stable_position,
                'predicted_position': (pred_x, pred_y),
                'velocity': self.velocity,
                'acceleration': self.acceleration,
                'elapsed_time': elapsed_time,
                'timestamp': time.time()
            })

        self.debug_info["predicted"] = True
        return (pred_x, pred_y)

    def _detect_backward_motion(self):
        """检测后退动作"""
        is_backward = False

        if not self.backward_enabled or len(self.position_history) < 3:
            return is_backward

        recent_positions = self.position_history.get_latest(3)
        y_changes = [recent_positions[i + 1][1] - recent_positions[i][1]
                     for i in range(len(recent_positions) - 1)]

        if all(dy > 0 for dy in y_changes) and sum(y_changes) > 10:
            is_backward = True
            self.backward_counter += 1
        else:
            self.backward_counter = max(0, self.backward_counter - 1)

        # 如果多次检测到后退动作，增强后退效果
        is_backward = is_backward or self.backward_counter >= 2

        # 发布后退状态事件
        if self.events and is_backward:
            self.events.publish("backward_motion_detected", {
                'counter': self.backward_counter,
                'timestamp': time.time()
            })

        return is_backward

    def update_motion_state(self, action, current_time):
        """更新运动状态"""
        previous_state = self.motion_state

        # 基于动作和遮挡状态更新
        if self.occlusion_detected:
            new_state = "occluded"
        elif action == "Moving":
            new_state = "moving"
        else:
            new_state = "static"

        # 状态变化或保持
        if new_state != previous_state:
            self.state_duration = 1
            self.motion_state = new_state

            # 发布状态变化事件
            if self.events:
                self.events.publish("motion_state_changed", {
                    'previous_state': previous_state,
                    'new_state': new_state,
                    'action': action,
                    'occlusion_detected': self.occlusion_detected,
                    'timestamp': current_time
                })
        else:
            self.state_duration += 1

            # 发布状态持续事件（间隔发送以减少事件量）
            if self.state_duration % 10 == 0 and self.events:
                self.events.publish("motion_state_continued", {
                    'state': self.motion_state,
                    'duration': self.state_duration,
                    'timestamp': current_time
                })

        # 更新调试信息
        self.debug_info["motion"] = self.motion_state

        return self.motion_state

    def adjust_kalman_parameters(self, motion_state, occlusion_counter=0,
                                 max_occlusion_frames=30):
        """根据运动状态动态调整卡尔曼滤波器参数"""
        if not self.kalman_filter:
            return

        # 保存旧参数用于比较
        if hasattr(self.kalman_filter, 'processNoiseCov'):
            old_process_noise = self.kalman_filter.processNoiseCov.copy()
        else:
            old_process_noise = None

        if hasattr(self.kalman_filter, 'measurementNoiseCov'):
            old_measurement_noise = self.kalman_filter.measurementNoiseCov.copy()
        else:
            old_measurement_noise = None

        # 基于状态选择基本参数
        if motion_state == "moving":
            # 移动状态 - 更快响应变化
            process_noise_pos = 0.06  # 位置过程噪声
            process_noise_vel = 0.12  # 速度过程噪声
            process_noise_acc = 0.18  # 加速度过程噪声
            measurement_noise = 0.05  # 测量噪声
        elif motion_state == "occluded":
            # 遮挡状态 - 进行渐进式调整
            occlusion_progress = min(1.0, occlusion_counter / (
                    max_occlusion_frames * 0.7))

            # 使用S型函数(sigmoid)实现平滑过渡
            sigmoid = 1.0 / (1.0 + np.exp(-6 * (occlusion_progress - 0.5)))

            # 参数随遮挡持续时间动态调整
            process_noise_pos = max(0.005, 0.04 * (1 - 0.7 * sigmoid))
            process_noise_vel = max(0.02, 0.08 * (1 - 0.6 * sigmoid))
            process_noise_acc = max(0.01, 0.08 * (1 - 0.8 * sigmoid))
            measurement_noise = 0.5 + 2.0 * sigmoid
        else:  # static
            # 静止状态 - 更稳定
            process_noise_pos = 0.005
            process_noise_vel = 0.01
            process_noise_acc = 0.02
            measurement_noise = 0.1

        # 创建过程噪声协方差矩阵
        self.kalman_filter.processNoiseCov = np.array([
            [process_noise_pos, 0, 0, 0, 0, 0],
            [0, process_noise_pos, 0, 0, 0, 0],
            [0, 0, process_noise_vel, 0, 0, 0],
            [0, 0, 0, process_noise_vel, 0, 0],
            [0, 0, 0, 0, process_noise_acc, 0],
            [0, 0, 0, 0, 0, process_noise_acc]
        ], np.float32)

        # 设置测量噪声协方差矩阵
        self.kalman_filter.measurementNoiseCov = np.array([
            [measurement_noise, 0],
            [0, measurement_noise]
        ], np.float32)

        # 检测参数是否有显著变化，发布事件
        significant_change = False

        if old_process_noise is not None:
            diff = np.abs(
                self.kalman_filter.processNoiseCov - old_process_noise).max()
            if diff > 0.005:
                significant_change = True

        if old_measurement_noise is not None:
            diff = np.abs(
                self.kalman_filter.measurementNoiseCov - old_measurement_noise).max()
            if diff > 0.1:
                significant_change = True

        if significant_change and self.events:
            self.events.publish("kalman_parameters_adjusted", {
                'motion_state': motion_state,
                'occlusion_progress': occlusion_counter / max_occlusion_frames if max_occlusion_frames > 0 else 0,
                'process_noise': {
                    'position': process_noise_pos,
                    'velocity': process_noise_vel,
                    'acceleration': process_noise_acc
                },
                'measurement_noise': measurement_noise,
                'timestamp': time.time()
            })

    def update_tracking(self, x, y, motion_state, occlusion_counter=0,
                        max_occlusion_frames=30):
        """使用新的测量值更新卡尔曼滤波器"""
        if not self.kalman_filter or not self.initialized:
            # 发布跟踪更新失败事件
            if self.events:
                self.events.publish("tracking_update_failed", {
                    'reason': 'not_initialized',
                    'position': (x, y),
                    'timestamp': time.time()
                })
            return None

        try:
            # 调整参数
            self.adjust_kalman_parameters(motion_state, occlusion_counter,
                                          max_occlusion_frames)

            # 预测
            prediction_start = time.time()
            prediction = self.kalman_filter.predict()
            prediction_time = time.time() - prediction_start

            # 准备测量值
            measurement = np.array([[float(x)], [float(y)]], np.float32)

            # 更新
            correction_start = time.time()
            corrected = self.kalman_filter.correct(measurement)
            correction_time = time.time() - correction_start

            # 提取位置
            kf_x = int(corrected[0, 0])
            kf_y = int(corrected[1, 0])

            # 确保位置在房间范围内
            kf_x = max(0, min(kf_x, self.room_width - 1))
            kf_y = max(0, min(kf_y, self.room_height - 1))

            # 发布跟踪更新成功事件
            if self.events:
                pred_x, pred_y = float(prediction[0, 0]), float(
                    prediction[1, 0])
                pred_vx, pred_vy = float(prediction[2, 0]), float(
                    prediction[3, 0])

                self.events.publish("tracking_updated", {
                    'input_position': (x, y),
                    'predicted_position': (pred_x, pred_y),
                    'corrected_position': (kf_x, kf_y),
                    'velocity': (pred_vx, pred_vy),
                    'prediction_time': prediction_time,
                    'correction_time': correction_time,
                    'motion_state': motion_state,
                    'timestamp': time.time()
                })

            return kf_x, kf_y

        except Exception as e:
            print(f"Kalman tracking error: {e}")

            # 发布跟踪错误事件
            if self.events:
                self.events.publish("tracking_error", {
                    'error': str(e),
                    'position': (x, y),
                    'motion_state': motion_state,
                    'timestamp': time.time()
                })

            return None

    def get_debug_info(self):
        """返回调试信息"""
        return self.debug_info

    def enable_debug_mode(self, enabled=True):
        """启用或禁用调试模式"""
        old_mode = self.debug_mode
        self.debug_mode = enabled

        # 发布调试模式变更事件
        if self.events and old_mode != enabled:
            self.events.publish("debug_mode_changed", {
                'old_mode': old_mode,
                'new_mode': enabled,
                'timestamp': time.time()
            })

        return self.debug_mode

    def enable_simple_mapping(self, enabled=True):
        """启用简单映射模式，用于测试"""
        old_mode = getattr(self, 'simple_mapping_mode', False)
        self.simple_mapping_mode = enabled
        print(f"Simple mapping mode {'enabled' if enabled else 'disabled'}")

        # 发布简单映射模式变更事件
        if self.events and old_mode != enabled:
            self.events.publish("simple_mapping_mode_changed", {
                'old_mode': old_mode,
                'new_mode': enabled,
                'timestamp': time.time()
            })

        return self.simple_mapping_mode

    def toggle_backward_enhancement(self, enabled=None):
        """开关后退增强功能"""
        old_state = self.backward_enabled

        if enabled is not None:
            self.backward_enabled = enabled
        else:
            self.backward_enabled = not self.backward_enabled

        # 发布后退增强功能变更事件
        if self.events and old_state != self.backward_enabled:
            self.events.publish("backward_enhancement_toggled", {
                'old_state': old_state,
                'new_state': self.backward_enabled,
                'timestamp': time.time()
            })

        return self.backward_enabled

    def toggle_feature(self, feature_name, state=None):
        """
        切换位置映射器特定功能

        Args:
            feature_name: 功能名称
            state: 要设置的状态 (True/False)，如果为None则切换当前状态

        Returns:
            bool: 是否成功切换功能
        """
        try:
            old_state = False

            # 支持后退功能增强的开关
            if feature_name == 'backward_enhancement':
                old_state = self.backward_enabled
                if state is None:
                    self.backward_enabled = not self.backward_enabled
                else:
                    self.backward_enabled = state
                new_state = self.backward_enabled

            # 支持简单映射模式的开关
            elif feature_name == 'simple_mapping':
                old_state = getattr(self, 'simple_mapping_mode', False)
                if state is None:
                    self.simple_mapping_mode = not old_state
                else:
                    self.simple_mapping_mode = state
                new_state = self.simple_mapping_mode
                print(f"Simple mapping mode {'enabled' if new_state else 'disabled'}")

            # 支持调试模式的开关
            elif feature_name == 'debug_mode':
                old_state = self.debug_mode
                if state is None:
                    self.debug_mode = not old_state
                else:
                    self.debug_mode = state
                new_state = self.debug_mode

            # 支持位置稳定化
            elif feature_name == 'stabilization':
                old_state = getattr(self, 'use_stabilization', False)
                if state is None:
                    self.use_stabilization = not old_state
                else:
                    self.use_stabilization = state
                new_state = self.use_stabilization

            # 支持平滑处理
            elif feature_name == 'smoothing':
                old_state = getattr(self, 'use_smoothing', False)
                if state is None:
                    self.use_smoothing = not old_state
                else:
                    self.use_smoothing = state
                new_state = self.use_smoothing

            else:
                # 不支持的功能
                print(f"位置映射器不支持功能: {feature_name}")

                # 发布不支持的功能事件
                if self.events:
                    self.events.publish("unsupported_feature", {
                        'feature': feature_name,
                        'timestamp': time.time()
                    })

                return False

            # 发布功能切换事件
            if self.events and old_state != new_state:
                self.events.publish("feature_toggled", {
                    'feature': feature_name,
                    'old_state': old_state,
                    'new_state': new_state,
                    'timestamp': time.time()
                })

            return True

        except Exception as e:
            # 记录错误
            print(f"切换位置映射器功能时出错: {e}")
            import traceback
            traceback.print_exc()

            # 发布功能切换错误事件
            if self.events:
                self.events.publish("feature_toggle_error", {
                    'feature': feature_name,
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False

    def get_feature_state(self, feature_name):
        """
        获取位置映射器特定功能的状态

        Args:
            feature_name: 功能名称

        Returns:
            bool: 功能当前状态
        """
        try:
            # 支持后退功能增强的开关
            if feature_name == 'backward_enhancement':
                state = getattr(self, 'backward_enabled', False)
            # 支持简单映射模式的开关
            elif feature_name == 'simple_mapping':
                state = getattr(self, 'simple_mapping_mode', False)
            # 支持调试模式的开关
            elif feature_name == 'debug_mode':
                state = getattr(self, 'debug_mode', False)
            # 支持位置稳定化
            elif feature_name == 'stabilization':
                state = getattr(self, 'use_stabilization', False)
            # 支持平滑处理
            elif feature_name == 'smoothing':
                state = getattr(self, 'use_smoothing', True)
            # 如果功能名称不受支持，返回False
            else:
                print(f"位置映射器不支持功能: {feature_name}")

                # 发布不支持的功能查询事件
                if self.events:
                    self.events.publish("unsupported_feature_query", {
                        'feature': feature_name,
                        'timestamp': time.time()
                    })

                return False

            # 发布功能状态查询事件
            if self.events:
                self.events.publish("feature_state_queried", {
                    'feature': feature_name,
                    'state': state,
                    'timestamp': time.time()
                })

            return state

        except Exception as e:
            print(f"获取位置映射器功能状态时出错: {e}")
            import traceback
            traceback.print_exc()

            # 发布功能状态查询错误事件
            if self.events:
                self.events.publish("feature_state_query_error", {
                    'feature': feature_name,
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False

    def get_kalman_state(self):
        """获取卡尔曼滤波器的当前状态向量"""
        if not self.kalman_filter or not self.initialized:
            return None

        try:
            state = self.kalman_filter.statePost
            return {
                'position': (float(state[0, 0]), float(state[1, 0])),
                'velocity': (float(state[2, 0]), float(state[3, 0])),
                'acceleration': (float(state[4, 0]), float(state[5, 0]))
            }
        except Exception as e:
            print(f"Error getting Kalman state: {e}")
            return None

    def get_velocity(self):
        """获取当前速度向量"""
        return self.velocity

    def get_acceleration(self):
        """获取当前加速度向量"""
        return self.acceleration

    def release_resources(self):
        """释放位置映射器使用的资源"""
        try:
            # 在这里添加释放资源的代码
            # 例如清理缓存、关闭文件等

            # 发布资源释放事件
            if hasattr(self, 'events') and self.events:
                self.events.publish("position_mapper_resources_released", {
                    'timestamp': time.time()
                })

            return True
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"释放位置映射器资源时出错: {e}")

            # 发布资源释放错误事件
            if hasattr(self, 'events') and self.events:
                self.events.publish("position_mapper_resource_release_error", {
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False
