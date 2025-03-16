import time
import logging
import psutil
import os
from utils.data_structures import CircularBuffer
from utils.logger_config import setup_logger, init_root_logger, setup_utf8_console


class SimplifiedSystemManager:
    """
    完整版系统管理器 - 整合系统状态管理、资源监控和性能监控功能
    提供一致的接口来监控系统状态、资源使用和性能统计
    """

    def __init__(self, log_interval=10, memory_warning_threshold=75,
                 memory_critical_threshold=85,event_system=None):
        """
        初始化完整版系统管理器

        Args:
            log_interval: 日志记录间隔（秒）
            memory_warning_threshold: 内存警告阈值（百分比）
            memory_critical_threshold: 内存临界阈值（百分比）
        """
        # 确保日志目录存在
        log_dir = "../.venv/logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 设置日志记录器
        log_file_path = os.path.join(log_dir, "system_manager.log")
        self.logger = logging.getLogger("SimplifiedSystemManager")
        self.logger = setup_logger("SimplifiedSystemManager", log_file_path)
        self.logger.info("SimplifiedSystemManager initialized")

        # ------------- 系统状态管理部分 -------------#
        # 系统状态参数
        self.system_state = "normal"  # normal, occlusion, recovery
        self.recovery_frames = 0
        self.max_recovery_frames = 10
        self.last_detection_time = 0
        self.detection_timeout = 0.5

        # 状态历史记录
        self.state_history = CircularBuffer(10)
        self.state_history.append(("normal", time.time()))

        # ------------- 资源监控部分 -------------#
        # 资源监控参数
        self.process = psutil.Process(os.getpid())
        self.last_check_time = 0
        self.log_interval = log_interval
        self.last_log_time = 0

        # 内存使用阈值
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold

        # 资源自适应策略状态
        self.adaptation_level = 0  # 0=正常, 1=警告, 2=临界
        self.adaptation_history = []
        self.last_adaptation_time = 0
        self.adaptation_cooldown = 30  # 资源自适应冷却时间（秒）

        # ------------- 性能监控部分 -------------#
        # 性能监控参数
        self.fps_history = []
        self.last_fps_time = time.time()
        self.fps = 0
        self.processing_times = {
            "detection": [],
            "action": [],
            "mapping": [],
            "total": []
        }
        self.max_samples = 100  # 最大样本数

        # 写入一条启动日志
        self.log_to_file(
            f"SimplifiedSystemManager initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 添加事件系统
        self.events = event_system

    # ===================== 系统状态管理方法 =====================#

    def update_state(self, persons_detected, current_time):
        """
        更新系统状态

        Args:
            persons_detected: 是否检测到人
            current_time: 当前时间戳

        Returns:
            str 或 None: 状态变化事件或None
        """
        event = None

        if persons_detected:
            self.last_detection_time = current_time

            # 使系统状态恢复正常
            if self.system_state == "occlusion":
                self.system_state = "recovery"
                self.recovery_frames = 0
                event = "recovery_started"
            elif self.system_state == "recovery":
                self.recovery_frames += 1
                if self.recovery_frames >= self.max_recovery_frames:
                    self.system_state = "normal"
                    event = "recovered"
        elif current_time - self.last_detection_time < self.detection_timeout:
            # 短暂检测失败
            if self.system_state == "normal":
                self.system_state = "occlusion"
                event = "occlusion_detected"
        else:
            # 长时间检测失败
            if self.system_state != "normal":
                old_state = self.system_state
                self.system_state = "normal"
                event = "reset"
                self.logger.info(f"系统状态重置: {old_state} -> normal")

        # 记录状态变化
        if event:
            self.state_history.append((self.system_state, current_time))
            self.logger.info(
                f"状态变化: {event}, 当前状态: {self.system_state}")

        return event

    def is_recent_detection(self, current_time):
        """
        检查是否最近有过检测

        Args:
            current_time: 当前时间戳

        Returns:
            bool: 是否最近有过检测
        """
        return current_time - self.last_detection_time < self.detection_timeout

    def get_state_duration(self):
        """
        获取当前状态持续时间

        Returns:
            float: 当前状态持续时间(秒)
        """
        if not self.state_history:
            return 0

        current_time = time.time()
        last_state, last_time = self.state_history[-1]
        return current_time - last_time

    def get_current_state(self):
        """
        获取当前系统状态

        Returns:
            str: 当前系统状态
        """
        return self.system_state

    # ===================== 资源监控方法 =====================#

    def get_memory_usage(self):
        """
        获取内存使用情况

        Returns:
            dict: 内存使用信息字典
        """
        try:
            # 更新检查时间
            self.last_check_time = time.time()

            # 获取系统内存使用
            system_memory = psutil.virtual_memory()
            system_percent = system_memory.percent
            system_available = system_memory.available / (
                    1024 * 1024 * 1024)  # GB

            # 获取进程内存使用
            process_memory = self.process.memory_info().rss / (
                    1024 * 1024)  # MB

            return {
                'system_percent': system_percent,
                'system_available_gb': system_available,
                'process_mb': process_memory,
                'timestamp': self.last_check_time
            }
        except Exception as e:
            self.logger.error(f"获取内存使用时出错: {e}")
            return None

    def get_cpu_usage(self):
        """
        获取CPU使用情况

        Returns:
            dict: CPU使用信息字典
        """
        try:
            # 系统CPU使用率
            system_cpu = psutil.cpu_percent(interval=0.1)

            # 进程CPU使用率
            process_cpu = self.process.cpu_percent(interval=0.1)

            # CPU核心数
            cpu_count = psutil.cpu_count(logical=True)

            return {
                'system_percent': system_cpu,
                'process_percent': process_cpu,
                'cpu_count': cpu_count,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"获取CPU使用时出错: {e}")
            return None

    def check_resources(self):
        """
        检查资源使用情况并确定当前资源状态

        Returns:
            dict: 资源状态信息
        """
        current_time = time.time()

        # 控制检查频率，避免过于频繁
        if current_time - self.last_check_time < 2.0:  # 至少2秒检查一次
            return {
                'level': self.adaptation_level,
                'status': 'cached'
            }

        # 获取内存使用情况
        memory_usage = self.get_memory_usage()
        if not memory_usage:
            return {
                'level': self.adaptation_level,
                'status': 'error'
            }

        system_percent = memory_usage['system_percent']
        process_mb = memory_usage['process_mb']

        # 获取CPU使用情况
        cpu_usage = self.get_cpu_usage()
        cpu_percent = cpu_usage['system_percent'] if cpu_usage else 0

        # 确定资源状态级别
        old_level = self.adaptation_level

        if system_percent > self.memory_critical_threshold or cpu_percent > 90:
            # 临界状态
            new_level = 2
        elif system_percent > self.memory_warning_threshold or cpu_percent > 75:
            # 警告状态
            new_level = 1
        else:
            # 正常状态
            new_level = 0

        # 如果状态变化，记录日志
        if new_level != old_level:
            level_names = ["正常", "警告", "临界"]
            self.logger.info(
                f"资源状态变化: {level_names[old_level]} -> {level_names[new_level]}")
            self.logger.info(
                f"内存使用: 系统 {system_percent:.1f}%, 进程 {process_mb:.1f}MB, CPU: {cpu_percent:.1f}%")

            # 记录状态变化
            self.adaptation_level = new_level
            self.adaptation_history.append((new_level, current_time))
            self.last_adaptation_time = current_time

        # 定期记录资源使用情况
        if current_time - self.last_log_time > self.log_interval:
            self.logger.info(
                f"资源使用: 内存 {system_percent:.1f}%, 进程 {process_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
            self.last_log_time = current_time

        return {
            'level': new_level,
            'memory': memory_usage,
            'cpu': cpu_usage,
            'status': 'updated'
        }

    def should_apply_adaptation(self):
        """
        检查是否应该应用资源自适应策略

        Returns:
            bool: 是否应该应用自适应策略
        """
        # 如果不处于警告或临界状态，不需要应用策略
        if self.adaptation_level == 0:
            return False

        # 如果距离上次应用策略的时间太短，等待冷却时间
        if time.time() - self.last_adaptation_time < self.adaptation_cooldown:
            return False

        return True

    def is_memory_critical(self):
        """
        检查内存是否达到临界值

        Returns:
            bool: 内存是否达到临界值
        """
        usage = self.get_memory_usage()
        if usage:
            return usage['system_percent'] > self.memory_critical_threshold
        return False

    def is_memory_warning(self):
        """
        检查内存是否达到警告值

        Returns:
            bool: 内存是否达到警告值
        """
        usage = self.get_memory_usage()
        if usage:
            return usage['system_percent'] > self.memory_warning_threshold
        return False

    def get_adaptation_level(self):
        """
        获取当前资源自适应级别

        Returns:
            int: 自适应级别 (0=正常, 1=警告, 2=临界)
        """
        return self.adaptation_level

    def get_adaptation_suggestions(self):
        """
        根据当前资源状态提供自适应建议

        Returns:
            dict: 自适应建议
        """
        suggestions = {
            'disable_features': [],
            'reduce_resolution': False,
            'increase_frame_interval': 0,
            'clear_history': False,
            'force_gc': False
        }

        if self.adaptation_level == 1:
            # 警告级别的建议
            suggestions['reduce_resolution'] = True
            suggestions['increase_frame_interval'] = 1

        elif self.adaptation_level == 2:
            # 临界级别的建议
            suggestions['disable_features'] = ['mediapipe', 'ml_model']
            suggestions['reduce_resolution'] = True
            suggestions['increase_frame_interval'] = 2
            suggestions['clear_history'] = True
            suggestions['force_gc'] = True

        return suggestions

    def reset_adaptation_state(self):
        """重置自适应状态"""
        self.adaptation_level = 0
        self.last_adaptation_time = 0
        self.adaptation_history = []

    # ===================== 性能监控方法 =====================#

    def start_timer(self):
        """
        开始计时

        Returns:
            float: 当前时间戳
        """
        return time.time()

    def record_time(self, category, start_time):
        """
        记录特定类别的处理时间

        Args:
            category: 时间类别 (detection, action, mapping, total)
            start_time: 开始时间戳

        Returns:
            float: 消耗的时间
        """
        elapsed = time.time() - start_time
        if category in self.processing_times:
            self.processing_times[category].append(elapsed)
            # 限制数组大小
            if len(self.processing_times[category]) > self.max_samples:
                self.processing_times[category] = self.processing_times[
                                                      category][
                                                  -self.max_samples:]

        # 定期记录性能日志
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            stats = self.get_performance_stats()
            if stats:
                self.logger.info(
                    f"性能统计: FPS={stats.get('estimated_fps', 0):.1f}, "
                    f"检测={stats.get('avg_detection', 0) * 1000:.1f}ms, "
                    f"动作={stats.get('avg_action', 0) * 1000:.1f}ms, "
                    f"映射={stats.get('avg_mapping', 0) * 1000:.1f}ms")
            self.last_log_time = current_time

        return elapsed

    def update_fps(self):
        """更新FPS计算"""
        current_time = time.time()
        time_diff = current_time - self.last_fps_time

        if time_diff >= 1.0:  # 每秒更新一次
            self.fps = len(self.fps_history) / time_diff
            self.fps_history = []
            self.last_fps_time = current_time
        else:
            self.fps_history.append(1)

    def get_performance_stats(self):
        """
        获取性能统计数据

        Returns:
            dict: 性能统计数据
        """
        if not self.processing_times["total"]:
            return None

        stats = {}
        for category in self.processing_times:
            if self.processing_times[category]:
                avg = sum(self.processing_times[category]) / len(
                    self.processing_times[category])
                stats[f"avg_{category}"] = avg

        if "avg_total" in stats and stats["avg_total"] > 0:
            stats["estimated_fps"] = 1.0 / stats["avg_total"]

        return stats

    def print_performance_stats(self):
        """打印性能统计数据"""
        stats = self.get_performance_stats()
        if not stats:
            return

        print("\n=== 性能统计 ===")
        print(
            f"平均帧处理时间: {stats['avg_total'] * 1000:.1f} ms (平均 FPS: {stats['estimated_fps']:.1f})")

        total = stats['avg_total']
        for category in ['detection', 'action', 'mapping']:
            if f"avg_{category}" in stats:
                avg = stats[f"avg_{category}"]
                percentage = (avg / total) * 100
                print(
                    f"{category.capitalize()}时间: {avg * 1000:.1f} ms ({percentage:.1f}%)")

        # 记录到日志
        self.log_to_file(
            f"性能统计: 总处理时间={stats['avg_total'] * 1000:.1f}ms, FPS={stats['estimated_fps']:.1f}")

    # ===================== 辅助方法 =====================#

    def log_to_file(self, message):
        """直接将消息写入日志文件"""
        try:
            log_dir = "../.venv/logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_file = os.path.join(log_dir, "system_manager_direct.log")
            with open(log_file, "a", encoding="utf-8") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - {message}\n")
        except Exception as e:
            print(f"写入日志文件失败: {e}")

    def on_config_changed(self, key, old_value, new_value):
        """
        响应配置系统的变更通知

        Args:
            key: 变更的配置键
            old_value: 变更前的值
            new_value: 变更后的值
        """
        self.logger.info(
            f"系统管理器配置变更: {key} = {new_value} (原值: {old_value})")

        try:
            # 处理各种配置项变更
            if key == "system.log_interval":
                self.log_interval = int(new_value)

            elif key == "system.memory_warning_threshold":
                self.memory_warning_threshold = float(new_value)

            elif key == "system.memory_critical_threshold":
                self.memory_critical_threshold = float(new_value)

            elif key == "system.async_mode":
                # 这个配置在TrackerApp级别处理，这里不做操作
                pass

            elif key == "system.performance_mode":
                # 例如，可以根据性能模式调整一些系统级设置
                # 此处为示例代码，可能需要根据实际情况调整
                if new_value == "high_speed":
                    self.detection_timeout = 0.3  # 更快检测超时
                elif new_value == "balanced":
                    self.detection_timeout = 0.5  # 默认检测超时
                elif new_value == "high_quality":
                    self.detection_timeout = 0.7  # 更长检测超时

            # 发布配置变更事件
            if hasattr(self, 'events') and self.events:
                self.events.publish("system_manager_config_changed", {
                    'key': key,
                    'old_value': old_value,
                    'new_value': new_value,
                    'timestamp': time.time()
                })

        except Exception as e:
            self.logger.error(f"应用系统管理器配置变更时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
