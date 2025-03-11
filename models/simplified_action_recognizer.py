import cv2
import time
import numpy as np

from utils.data_structures import CircularBuffer
from utils.cache_utils import create_standard_cache, generate_cache_key
from interface.action_recognizer_interface import ActionRecognizerInterface


class SimplifiedActionRecognizer(ActionRecognizerInterface):
    """
    简化版的动作识别器，整合了多个功能模块
    使用组合模式代替多继承，提高代码清晰度和性能
    集成事件系统，发布动作识别的关键事件
    """

    def __init__(self, config=None, event_system=None):
        """
        初始化动作识别器

        Args:
            config: 配置字典，可选
            event_system: 事件系统实例，可选
        """
        # 保存事件系统
        self.events = event_system

        # 默认配置
        self.default_config = {
            'keypoint_confidence_threshold': 0.5,
            'position_movement_threshold': 10,
            'motion_cooldown': 0.5,
            'history_length': 10,
            'action_history_length': 5,
            'waving_threshold': 50,
            'jumping_threshold': 30,
            'moving_threshold': 10,
            'frame_interval': 3,  # 每3帧分析一次
        }

        # 初始化缓存
        self.cache_hits = 0
        self.cache_misses = 0

        self.action_cache = create_standard_cache(
            name="action_recognizer",
            capacity=25,
            timeout=0.5
        )

        # 合并用户配置
        self.config = {**self.default_config, **(config or {})}

        # 初始化历史数据
        self.keypoints_history = CircularBuffer(self.config['history_length'])
        self.action_history = CircularBuffer(
            self.config['action_history_length'])
        self.position_history = CircularBuffer(self.config['history_length'])

        # 记录时间
        self.last_motion_time = time.time()
        self.motion_cooldown = self.config['motion_cooldown']

        # 参数设置
        self.keypoint_confidence_threshold = self.config[
            'keypoint_confidence_threshold']
        self.position_movement_threshold = self.config[
            'position_movement_threshold']

        # 帧计数
        self.frame_count = 0
        self.latest_result = "Static"

        # 为插件系统预留钩子点
        self.plugin_hooks = {
            "pre_process": [],
            "feature_extraction": [],
            "recognition": [],
            "post_process": []
        }

        self.current_state = "static"

        # 发布初始化完成事件
        if self.events:
            self.events.publish("action_recognizer_initialized", {
                'config': self.config,
                'timestamp': time.time()
            })

    def analyze_wrist_movement(self, keypoints_sequence):
        """
        分析手腕移动幅度

        Args:
            keypoints_sequence: 关键点序列

        Returns:
            float: 手腕最大移动幅度
        """
        # 导入距离计算函数
        from utils import MathUtils

        # 手腕关键点索引 (通常是9和10)
        wrist_indices = [9, 10]  # 左右手腕

        max_movement = 0
        for idx in wrist_indices:
            # 提取手腕位置序列
            positions = []
            for kp in keypoints_sequence:
                if len(kp) > idx and kp[idx][
                    2] >= self.keypoint_confidence_threshold:
                    positions.append((kp[idx][0], kp[idx][1]))

            if len(positions) < 2:
                continue

            # 计算最大移动距离
            max_dist = 0
            for i in range(len(positions) - 1):
                dist = MathUtils.distance(positions[i], positions[i + 1])
                if dist > max_dist:
                    max_dist = dist

            max_movement = max(max_movement, max_dist)

        # 发布手腕移动分析事件
        if self.events and max_movement > 0:
            self.events.publish("wrist_movement_analyzed", {
                'max_movement': max_movement,
                'waving_threshold': self.config.get('waving_threshold', 50),
                'timestamp': time.time()
            })

        return max_movement

    def analyze_vertical_movement(self, keypoints_sequence):
        """
        分析垂直移动幅度

        Args:
            keypoints_sequence: 关键点序列

        Returns:
            float: 垂直移动幅度
        """
        # 使用鼻子或颈部关键点 (通常是0)
        nose_idx = 0

        # 提取鼻子位置序列
        positions = []
        for kp in keypoints_sequence:
            if len(kp) > nose_idx and kp[nose_idx][
                2] >= self.keypoint_confidence_threshold:
                positions.append(kp[nose_idx][1])  # 只取y坐标

        if len(positions) < 2:
            return 0

        # 计算最大最小值
        min_pos = min(positions)
        max_pos = max(positions)

        vertical_movement = max_pos - min_pos

        # 发布垂直移动分析事件
        if self.events and vertical_movement > 0:
            self.events.publish("vertical_movement_analyzed", {
                'vertical_movement': vertical_movement,
                'jumping_threshold': self.config.get('jumping_threshold', 30),
                'min_y': min_pos,
                'max_y': max_pos,
                'timestamp': time.time()
            })

        return vertical_movement

    def analyze_position_change(self, position_history):
        """
        分析位置变化

        Args:
            position_history: 位置历史记录

        Returns:
            float: 平均位移速度
        """
        if len(position_history) < 3:
            return 0

        # 导入距离计算函数
        from utils import MathUtils

        # 只取最近的3个位置
        positions = list(position_history)[-3:]

        # 计算平均位移速度
        total_distance = 0
        for i in range(len(positions) - 1):
            total_distance += MathUtils.distance(positions[i], positions[i + 1])

        avg_speed = total_distance / (len(positions) - 1)

        # 发布位置变化分析事件
        if self.events and avg_speed > 0:
            self.events.publish("position_change_analyzed", {
                'average_speed': avg_speed,
                'moving_threshold': self.config.get('moving_threshold', 10),
                'is_moving': avg_speed > self.position_movement_threshold,
                'timestamp': time.time()
            })

        return avg_speed

    def _recognize_with_rules(self, features, current_time):
        """
        使用规则引擎识别动作

        Args:
            features: 特征字典
            current_time: 当前时间

        Returns:
            str: 识别的动作
        """
        previous_state = self.current_state

        # 最高优先级：移动检测
        if features['is_moving']:
            self.current_state = "moving"
            self.last_motion_time = current_time
            action_str = "Moving"

        # 次优先级：垂直移动（跳跃）
        elif features['vertical_movement'] > self.config.get(
                'jumping_threshold', 30):
            self.current_state = "jumping"
            self.last_motion_time = current_time
            action_str = "Jumping"

        # 再次优先级：手腕移动（挥手）
        elif features['wrist_movement'] > self.config.get('waving_threshold',
                                                          50):
            self.current_state = "waving"
            self.last_motion_time = current_time
            action_str = "Waving"

        # 最低优先级：静止状态
        else:
            # 如果离上次动作检测超过2秒，恢复静止状态
            if current_time - self.last_motion_time > 2:
                self.current_state = "static"
            action_str = self.current_state.capitalize()

        # 发布状态变化事件
        if self.events and previous_state != self.current_state:
            self.events.publish("action_state_changed", {
                'previous_state': previous_state,
                'new_state': self.current_state,
                'action_string': action_str,
                'timestamp': current_time
            })

        return action_str

    def _smooth_with_history(self, current_action):
        """
        使用历史数据平滑动作识别结果

        Args:
            current_action: 当前识别的动作

        Returns:
            str: 平滑后的动作
        """
        # 如果历史记录不够，直接返回当前结果
        if len(self.action_history) < 3:
            return current_action

        # 计算动作频率
        counts = {}
        for act in self.action_history:
            counts[act] = counts.get(act, 0) + 1
        counts[current_action] = counts.get(current_action, 0) + 1

        # 找出最多的动作
        max_count = 0
        most_common = current_action
        for act, count in counts.items():
            if count > max_count:
                max_count = count
                most_common = act

        # 只有当最常见动作达到一定比例时才更新
        threshold = 0.4  # 40%的阈值
        before_smoothing = current_action
        after_smoothing = most_common
        if max_count >= (len(self.action_history) + 1) * threshold:
            after_smoothing = most_common
        else:
            after_smoothing = current_action

        # 发布动作平滑事件
        if self.events and before_smoothing != after_smoothing:
            self.events.publish("action_smoothed", {
                'before_smoothing': before_smoothing,
                'after_smoothing': after_smoothing,
                'max_count': max_count,
                'threshold': threshold,
                'history_length': len(self.action_history),
                'timestamp': time.time()
            })

        return after_smoothing

    def _calculate_features(self, person):
        """
        计算特征

        Args:
            person: 人体信息字典

        Returns:
            dict: 特征字典
        """
        # 获取最新的几帧关键点
        recent_keypoints = self.keypoints_history.get_latest(3)

        # 计算特征
        features = {}

        # 1. 计算位置变化
        position_change = self.analyze_position_change(self.position_history)
        features['position_change'] = position_change
        features[
            'is_moving'] = position_change > self.position_movement_threshold

        # 2. 如果不是移动状态，计算其他特征
        if not features['is_moving']:
            features['wrist_movement'] = self.analyze_wrist_movement(
                recent_keypoints)
            features['vertical_movement'] = self.analyze_vertical_movement(
                recent_keypoints)
        else:
            # 如果是移动，设置默认值避免不必要的计算
            features['wrist_movement'] = 0
            features['vertical_movement'] = 0

        # 3. 计算高度变化（如果有校准数据）
        if 'calibration_height' in person and 'height' in person:
            features['height_change'] = person['height'] / person[
                'calibration_height'] - 1.0
        else:
            features['height_change'] = 0

        # 发布特征计算事件
        if self.events:
            self.events.publish("features_calculated", {
                'position_change': features['position_change'],
                'is_moving': features['is_moving'],
                'wrist_movement': features['wrist_movement'],
                'vertical_movement': features['vertical_movement'],
                'height_change': features['height_change'],
                'timestamp': time.time()
            })

        return features

    def recognize_action(self, person):
        """
        对外暴露的主要接口：识别动作

        Args:
            person: 人体信息字典

        Returns:
            str: 识别的动作
        """
        recognition_start_time = time.time()

        # 帧率控制 - 降低处理频率
        self.frame_count += 1
        if (self.frame_count % self.config['frame_interval'] != 0 and
                self.frame_count > self.config['frame_interval']):
            return self.latest_result

        # 创建缓存键 - 使用关键点的哈希值
        if 'keypoints' in person:
            # 只使用高置信度关键点的位置
            key_points = []
            for kp in person['keypoints']:
                if kp[2] > self.keypoint_confidence_threshold:
                    key_points.append((int(kp[0]), int(kp[1])))

            # 使用新的键生成函数
            cache_key = generate_cache_key(key_points, prefix="action")

            # 检查缓存
            cached_action = self.action_cache.get(cache_key)
            if cached_action is not None:
                self.cache_hits += 1
                hit_rate = self.cache_hits / (
                        self.cache_hits + self.cache_misses)

                # 发布缓存命中事件
                if self.events and self.cache_hits % 10 == 0:  # 每10次命中发送一次
                    self.events.publish("action_cache_hit", {
                        'hit_rate': hit_rate,
                        'cache_hits': self.cache_hits,
                        'cache_misses': self.cache_misses,
                        'action': cached_action,
                        'timestamp': time.time()
                    })

                if self.cache_hits % 20 == 0:
                    print(f"动作识别缓存命中率: {hit_rate:.2f}")
                return cached_action

            self.cache_misses += 1

            # 发布缓存未命中事件
            if self.events:
                self.events.publish("action_cache_miss", {
                    'timestamp': time.time()
                })

        # 收集数据 - 每帧都做，确保历史连续
        self._collect_data(person)

        # 如果历史不够长，无法分析
        if len(self.keypoints_history) < 3:
            # 发布历史数据不足事件
            if self.events:
                self.events.publish("insufficient_history", {
                    'current_history': len(self.keypoints_history),
                    'required_history': 3,
                    'timestamp': time.time()
                })

            return "Collecting data..."

        # 检查冷却时间 - 防止动作抖动
        current_time = time.time()
        if current_time - self.last_motion_time < self.motion_cooldown:
            # 发布冷却中事件
            if self.events:
                self.events.publish("motion_cooldown_active", {
                    'remaining_cooldown': self.motion_cooldown - (
                            current_time - self.last_motion_time),
                    'cooldown_period': self.motion_cooldown,
                    'timestamp': current_time
                })

            return self.latest_result

        # 运行前处理插件钩子
        person_after_preprocess = self._run_plugin_hooks("pre_process", person)

        # 计算特征
        features = self._calculate_features(person_after_preprocess)

        # 运行特征提取插件钩子
        features_enhanced = self._run_plugin_hooks("feature_extraction",
                                                   features)

        # 使用规则引擎识别动作
        result = self._recognize_with_rules(features_enhanced, current_time)

        # 运行识别插件钩子
        result_after_recognition = self._run_plugin_hooks("recognition", result)

        # 平滑结果
        result = self._smooth_with_history(result_after_recognition)

        # 运行后处理插件钩子
        final_result = self._run_plugin_hooks("post_process", result)

        # 更新动作历史
        self.action_history.append(final_result)

        # 更新最新结果
        previous_result = self.latest_result
        self.latest_result = final_result

        # 如果不是静止状态，更新动作时间
        if final_result != "Static":
            self.last_motion_time = current_time

        # 缓存结果
        if 'keypoints' in person and final_result:
            self.action_cache.put(cache_key, final_result)
            # 添加调试信息
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"缓存动作结果: {final_result}, 键: {cache_key[:8]}...")

            # 动作变化时发布事件
            if self.events and final_result != previous_result:
                recognition_time = time.time() - recognition_start_time
                self.events.publish("action_recognized", {
                    'person': person,
                    'action': final_result,
                    'previous_action': previous_result,
                    'features': features,
                    'recognition_time': recognition_time,
                    'timestamp': time.time()
                })

        # 发布处理完成事件
        if self.events:
            self.events.publish("action_recognition_completed", {
                'action': final_result,
                'processing_time': time.time() - recognition_start_time,
                'frame_count': self.frame_count,
                'timestamp': time.time()
            })

        return final_result

    def _collect_data(self, person):
        """
        收集数据到历史记录

        Args:
            person: 人体信息字典
        """
        # 保存关键点历史
        if 'keypoints' in person:
            self.keypoints_history.append(person['keypoints'])

            # 发布数据收集事件
            if self.events:
                self.events.publish("keypoints_collected", {
                    'keypoints_count': len(person['keypoints']),
                    'history_length': len(self.keypoints_history),
                    'timestamp': time.time()
                })

        # 保存位置历史
        if 'center_x' in person and 'center_y' in person:
            current_pos = (person['center_x'], person['center_y'])
            self.position_history.append(current_pos)

            # 发布位置收集事件
            if self.events:
                self.events.publish("position_collected", {
                    'position': current_pos,
                    'history_length': len(self.position_history),
                    'timestamp': time.time()
                })

    def _run_plugin_hooks(self, hook_name, data):
        """
        运行指定钩子点的所有插件

        Args:
            hook_name: 钩子点名称
            data: 输入数据

        Returns:
            任意类型: 处理后的数据
        """
        if hook_name not in self.plugin_hooks:
            return data

        result = data
        for plugin_func in self.plugin_hooks[hook_name]:
            try:
                plugin_start_time = time.time()
                result = plugin_func(result)
                plugin_time = time.time() - plugin_start_time

                # 发布插件执行事件
                if self.events:
                    self.events.publish("plugin_executed", {
                        'hook_name': hook_name,
                        'plugin_name': getattr(plugin_func, '__name__',
                                               'unknown'),
                        'execution_time': plugin_time,
                        'timestamp': time.time()
                    })

            except Exception as e:
                print(f"Plugin error at {hook_name}: {e}")

                # 发布插件错误事件
                if self.events:
                    self.events.publish("plugin_error", {
                        'hook_name': hook_name,
                        'plugin_name': getattr(plugin_func, '__name__',
                                               'unknown'),
                        'error': str(e),
                        'timestamp': time.time()
                    })

        return result

    def get_feature_state(self, feature_name):
        """
        获取动作识别器特定功能的状态

        Args:
            feature_name: 功能名称

        Returns:
            bool: 功能当前状态
        """
        try:
            if feature_name == 'ml_model':
                state = self.config.get('enable_ml_model', False) and hasattr(
                    self, 'ml_recognizer') and self.ml_recognizer is not None
            elif feature_name == 'dtw':
                state = self.config.get('enable_dtw', False) and hasattr(self,
                                                                         'dtw_recognizer') and self.dtw_recognizer is not None
            elif feature_name == 'threading':
                state = self.config.get('enable_threading', False)
            else:
                print(f"动作识别器不支持功能: {feature_name}")

                # 发布功能查询错误事件
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
            print(f"获取动作识别器功能状态时出错: {e}")

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
        切换动作识别器特定功能

        Args:
            feature_name: 功能名称 (例如 'ml_model', 'dtw', 'threading')
            state: 要设置的状态 (True/False)

        Returns:
            bool: 是否成功切换功能
        """
        try:
            if feature_name == 'ml_model':
                # 更新配置
                old_state = self.config.get('enable_ml_model', False)
                self.config['enable_ml_model'] = state

                # 应用更改
                if state:
                    if not hasattr(self, 'ml_recognizer') or self.ml_recognizer is None:
                        # 尝试导入或创建ML识别器
                        try:
                            from advanced_recognition import MLActionRecognizer
                            self.ml_recognizer = MLActionRecognizer(self.config)
                            success = True
                        except ImportError:
                            # 尝试使用内部类(如果有)
                            if hasattr(self, 'MLActionRecognizer'):
                                self.ml_recognizer = self.MLActionRecognizer(self.config)
                                success = True
                            else:
                                print("无法导入MLActionRecognizer")
                                success = False
                    else:
                        success = True
                else:
                    # 禁用ML模型
                    self.ml_recognizer = None
                    success = True

                print(f"ML模型功能已{'启用' if state else '禁用'}")

                # 发布功能切换事件
                if self.events:
                    self.events.publish("feature_toggled", {
                        'feature_name': feature_name,
                        'old_state': old_state,
                        'new_state': state,
                        'success': success,
                        'timestamp': time.time()
                    })

                return success

            elif feature_name == 'dtw':
                # 更新配置
                old_state = self.config.get('enable_dtw', False)
                self.config['enable_dtw'] = state

                # 应用更改
                if state:
                    if not hasattr(self, 'dtw_recognizer') or self.dtw_recognizer is None:
                        # 尝试导入或创建DTW识别器
                        try:
                            from advanced_recognition import DTWActionRecognizer
                            self.dtw_recognizer = DTWActionRecognizer(self.config)
                            success = True
                        except ImportError:
                            # 尝试使用内部类(如果有)
                            if hasattr(self, 'DTWActionRecognizer'):
                                self.dtw_recognizer = self.DTWActionRecognizer(self.config)
                                success = True
                            else:
                                print("无法导入DTWActionRecognizer")
                                success = False
                    else:
                        success = True
                else:
                    # 禁用DTW
                    self.dtw_recognizer = None
                    success = True

                print(f"DTW功能已{'启用' if state else '禁用'}")

                # 发布功能切换事件
                if self.events:
                    self.events.publish("feature_toggled", {
                        'feature_name': feature_name,
                        'old_state': old_state,
                        'new_state': state,
                        'success': success,
                        'timestamp': time.time()
                    })

                return success

            elif feature_name == 'threading':
                # 记录当前值以便恢复
                old_value = self.config.get('enable_threading', False)

                # 设置新值
                self.config['enable_threading'] = state

                # 如果开启多线程，确保线程池可用
                if state:
                    try:
                        from utils import get_thread_pool
                        # 尝试获取线程池，以验证线程池可以初始化
                        thread_pool = get_thread_pool(self.config.get('max_workers', 2))
                        success = True
                    except Exception as e:
                        print(f"初始化线程池失败: {e}")
                        # 恢复旧值
                        self.config['enable_threading'] = old_value
                        success = False
                else:
                    success = True

                print(f"多线程功能已{'启用' if state else '禁用'}")

                # 发布功能切换事件
                if self.events:
                    self.events.publish("feature_toggled", {
                        'feature_name': feature_name,
                        'old_state': old_value,
                        'new_state': state,
                        'success': success,
                        'timestamp': time.time()
                    })

                return success

            else:
                print(f"动作识别器不支持功能: {feature_name}")

                # 发布不支持的功能事件
                if self.events:
                    self.events.publish("unsupported_feature_toggle", {
                        'feature_name': feature_name,
                        'timestamp': time.time()
                    })

                return False

        except Exception as e:
            print(f"切换动作识别器功能时出错: {e}")
            import traceback
            traceback.print_exc()

            # 发布功能切换错误事件
            if self.events:
                self.events.publish("feature_toggle_error", {
                    'feature_name': feature_name,
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False

    def release_resources(self):
        """
        释放识别器使用的资源

        Returns:
            bool: 是否成功释放资源
        """
        try:
            # 清空历史数据
            self.keypoints_history.clear()
            self.position_history.clear()
            self.action_history.clear()

            # 清理缓存
            if hasattr(self, 'action_cache'):
                cache_stats = {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_rate': self.cache_hits / (
                            self.cache_hits + self.cache_misses) if (
                                                                            self.cache_hits + self.cache_misses) > 0 else 0
                }

                self.action_cache.clear()
                print("动作识别缓存已清理")

            # 清空插件钩子
            plugins_count = sum(
                len(hooks) for hooks in self.plugin_hooks.values())
            for hook in self.plugin_hooks:
                self.plugin_hooks[hook] = []

            # 强制垃圾回收
            import gc
            gc.collect()

            print("动作识别器资源已释放")

            # 发布资源释放事件
            if self.events:
                self.events.publish("resources_released", {
                    'cache_stats': cache_stats if 'cache_stats' in locals() else {},
                    'plugins_count': plugins_count if 'plugins_count' in locals() else 0,
                    'timestamp': time.time()
                })

            return True
        except Exception as e:
            print(f"释放动作识别器资源时出错: {e}")
            import traceback
            traceback.print_exc()

            # 发布资源释放错误事件
            if self.events:
                self.events.publish("resource_release_error", {
                    'error': str(e),
                    'timestamp': time.time()
                })

            return False
