# utils/event_logger.py

import os
import json
import time
import logging
import threading
from datetime import datetime


class EventLogger:
    """
    事件日志记录器 - 用于记录事件流和持久化事件

    这个类提供自动事件日志记录和定期事件持久化功能，
    它会监听所有事件，并定期将事件保存到文件中。
    """

    def __init__(self, event_system, log_dir="logs/events",
                 auto_save_interval=300, max_events_per_file=1000):
        """
        初始化事件日志记录器

        Args:
            event_system: 事件系统实例
            log_dir: 事件日志保存目录
            auto_save_interval: 自动保存间隔(秒)
            max_events_per_file: 每个文件最大事件数量
        """
        self.event_system = event_system
        self.log_dir = log_dir
        self.auto_save_interval = auto_save_interval
        self.max_events_per_file = max_events_per_file

        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 设置日志记录器
        self.logger = logging.getLogger("EventLogger")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 当前事件集
        self.current_events = []
        self.last_save_time = time.time()

        # 线程安全锁
        self.lock = threading.RLock()

        # 注册事件监听器
        self._register_event_listeners()

        self.logger.info(f"事件日志记录器已初始化，保存目录: {log_dir}")

    def _register_event_listeners(self):
        """注册全局事件监听器来捕获所有事件"""
        # 在事件系统中订阅所有事件类型
        # 如果事件系统支持通配符，可以使用 "*" 订阅所有事件
        if hasattr(self.event_system, 'subscribe_wildcard'):
            self.event_system.subscribe_wildcard(self._on_any_event)
            self.logger.info("已使用通配符订阅所有事件")
        else:
            # 如果不支持通配符，我们需要手动订阅一些关键事件类型
            # 这里只是示例，实际应用中可能需要订阅更多事件类型
            common_events = [
                "system_event", "user_event", "app_event",
                "person_detected", "action_recognized", "position_mapped",
                "feature_toggled", "key_pressed", "ui_updated"
            ]

            for event_type in common_events:
                self.event_system.subscribe(event_type, self._on_specific_event)

            self.logger.info(f"已订阅 {len(common_events)} 种常见事件类型")

    def _on_specific_event(self, data):
        """处理特定类型的事件"""
        event_type = data.get('event_type', 'unknown')
        self._record_event(event_type, data)

    def _on_any_event(self, event_type, data):
        """处理任何类型的事件"""
        self._record_event(event_type, data)

    def _record_event(self, event_type, data):
        """记录事件到当前集合"""
        with self.lock:
            # 将事件添加到当前集合
            self.current_events.append((event_type, data))

            # 检查是否需要自动保存
            current_time = time.time()
            if (current_time - self.last_save_time >= self.auto_save_interval or
                    len(self.current_events) >= self.max_events_per_file):
                self.save_current_events()

    def save_current_events(self):
        """保存当前收集的事件到文件"""
        with self.lock:
            if not self.current_events:
                return 0

            try:
                # 生成文件名: events_YYYYMMDD_HHMMSS.json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"events_{timestamp}.json"
                filepath = os.path.join(self.log_dir, filename)

                # 序列化事件
                from utils.event_serializer import serialize_event
                serialized_events = [
                    serialize_event(event_type, event_data)
                    for event_type, event_data in self.current_events
                ]

                # 保存到文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(serialized_events, f, indent=2)

                event_count = len(self.current_events)
                self.logger.info(f"已保存 {event_count} 个事件到 {filepath}")

                # 清空当前事件集
                self.current_events = []
                self.last_save_time = time.time()

                return event_count
            except Exception as e:
                self.logger.error(f"保存事件失败: {e}")
                return 0

    def stop(self):
        """停止事件记录并保存剩余事件"""
        # 如果有通配符订阅，取消它
        if hasattr(self.event_system, 'unsubscribe_wildcard'):
            self.event_system.unsubscribe_wildcard(self._on_any_event)
        else:
            # 如果不支持通配符，取消所有特定类型的订阅
            # 这里假设我们知道之前订阅了哪些事件类型
            common_events = [
                "system_event", "user_event", "app_event",
                "person_detected", "action_recognized", "position_mapped",
                "feature_toggled", "key_pressed", "ui_updated"
            ]

            for event_type in common_events:
                self.event_system.unsubscribe(event_type,
                                              self._on_specific_event)

        # 保存剩余事件
        saved = self.save_current_events()
        self.logger.info(f"事件日志记录器已停止，最后保存了 {saved} 个事件")
