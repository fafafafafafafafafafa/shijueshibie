#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置变更通知系统 - 管理配置变更通知和处理

该模块实现了配置变更的观察者模式，允许系统组件订阅特定配置的变更，
并在配置变更时接收通知。它支持精细的订阅控制和批量通知处理。
"""

import logging
import threading
import time
from typing import Any, Dict, List, Set, Callable, Optional, Tuple, Union, \
    Pattern
import re
from enum import Enum, auto
import queue
import weakref
from collections import defaultdict

logger = logging.getLogger(__name__)


class NotificationMode(Enum):
    """配置通知模式枚举"""
    IMMEDIATE = auto()  # 立即通知
    BATCH = auto()  # 批量通知
    THROTTLE = auto()  # 限流通知
    DEBOUNCE = auto()  # 去抖动通知


class SubscriptionType(Enum):
    """订阅类型枚举"""
    EXACT = auto()  # 精确匹配
    PREFIX = auto()  # 前缀匹配
    PATTERN = auto()  # 模式匹配
    CATEGORY = auto()  # 分类匹配
    ALL = auto()  # 所有配置


class ConfigChangeEvent:
    """配置变更事件类"""

    def __init__(self, key: str, old_value: Any, new_value: Any,
                 timestamp: float = None, source: str = None):
        """初始化配置变更事件

        Args:
            key: 配置键
            old_value: 旧值
            new_value: 新值
            timestamp: 时间戳
            source: 事件来源
        """
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or time.time()
        self.source = source

    def __repr__(self) -> str:
        """返回事件的字符串表示"""
        return f"ConfigChangeEvent(key='{self.key}', old={self.old_value}, new={self.new_value})"

    def merge(self, other: 'ConfigChangeEvent') -> 'ConfigChangeEvent':
        """合并另一个事件（用于批处理）

        Args:
            other: 另一个事件

        Returns:
            ConfigChangeEvent: 合并后的事件
        """
        # 保留最新的时间戳和值
        if other.timestamp > self.timestamp:
            return ConfigChangeEvent(
                key=self.key,
                old_value=self.old_value,  # 保留原始旧值
                new_value=other.new_value,  # 使用最新的新值
                timestamp=other.timestamp,
                source=other.source
            )
        return self


class ConfigSubscriber:
    """配置订阅者类"""

    def __init__(
            self,
            subscriber_id: str,
            callback: Callable[[ConfigChangeEvent], None],
            subscription_type: SubscriptionType,
            subscription_filter: Any,
            priority: int = 0,
            throttle_interval: float = 0.0,
            debounce_interval: float = 0.0
    ):
        """初始化配置订阅者

        Args:
            subscriber_id: 订阅者ID
            callback: 回调函数
            subscription_type: 订阅类型
            subscription_filter: 订阅过滤器
            priority: 优先级
            throttle_interval: 限流间隔（秒）
            debounce_interval: 去抖动间隔（秒）
        """
        self.id = subscriber_id
        self.callback = callback
        self.type = subscription_type
        self.filter = subscription_filter
        self.priority = priority
        self.throttle_interval = throttle_interval
        self.debounce_interval = debounce_interval
        self.last_notification_time = 0.0
        self.pending_event = None
        self.debounce_timer = None

    def matches(self, key: str,
                categories: Optional[Dict[str, List[str]]] = None) -> bool:
        """检查键是否匹配订阅条件

        Args:
            key: 配置键
            categories: 配置分类映射

        Returns:
            bool: 是否匹配
        """
        if self.type == SubscriptionType.EXACT:
            return key == self.filter

        elif self.type == SubscriptionType.PREFIX:
            return key.startswith(self.filter)

        elif self.type == SubscriptionType.PATTERN:
            if isinstance(self.filter, str):
                # 编译正则表达式
                self.filter = re.compile(self.filter)
            return bool(self.filter.match(key))

        elif self.type == SubscriptionType.CATEGORY:
            if not categories:
                return False
            category_keys = categories.get(self.filter, [])
            return key in category_keys

        elif self.type == SubscriptionType.ALL:
            return True

        return False

    def should_notify(self, current_time: float) -> bool:
        """判断是否应该通知订阅者（用于限流）

        Args:
            current_time: 当前时间

        Returns:
            bool: 是否应该通知
        """
        # 如果没有设置限流间隔，则总是通知
        if self.throttle_interval <= 0:
            return True

        # 检查是否已经过了限流间隔
        return (
                    current_time - self.last_notification_time) >= self.throttle_interval

    def notify(self, event: ConfigChangeEvent) -> None:
        """通知订阅者配置变更

        Args:
            event: 配置变更事件
        """
        try:
            self.last_notification_time = time.time()
            self.callback(event)
        except Exception as e:
            logger.error(
                f"Error notifying subscriber {self.id} about {event.key}: {str(e)}")

    def schedule_debounced_notification(self, event: ConfigChangeEvent) -> None:
        """安排去抖动通知

        Args:
            event: 配置变更事件
        """
        # 取消现有的定时器
        if self.debounce_timer:
            self.debounce_timer.cancel()

        # 保存最新的事件
        self.pending_event = event

        # 创建新的定时器
        self.debounce_timer = threading.Timer(
            self.debounce_interval,
            self._execute_debounced_notification
        )
        self.debounce_timer.daemon = True
        self.debounce_timer.start()

    def _execute_debounced_notification(self) -> None:
        """执行去抖动通知"""
        if self.pending_event:
            self.notify(self.pending_event)
            self.pending_event = None

    def cancel_pending_notifications(self) -> None:
        """取消待处理的通知"""
        if self.debounce_timer:
            self.debounce_timer.cancel()
            self.debounce_timer = None
        self.pending_event = None

    def __repr__(self) -> str:
        """返回订阅者的字符串表示"""
        return f"ConfigSubscriber(id='{self.id}', type={self.type}, filter={self.filter})"


class ConfigChangeNotifier:
    """配置变更通知器类 - 管理配置变更通知和订阅"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigChangeNotifier, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, batch_interval: float = 0.1,
                 enable_batch_mode: bool = False):
        """初始化配置变更通知器

        Args:
            batch_interval: 批处理间隔（秒）
            enable_batch_mode: 是否启用批处理模式
        """
        # 避免重复初始化
        with self._lock:
            if self._initialized:
                return

            # 订阅者注册表，格式：{subscriber_id: ConfigSubscriber}
            self._subscribers: Dict[str, ConfigSubscriber] = {}

            # 配置分类映射
            self._categories: Dict[str, List[str]] = {}

            # 启用批处理模式
            self._batch_mode = enable_batch_mode
            self._batch_interval = batch_interval
            self._batch_queue = queue.Queue()
            self._batch_thread = None
            self._should_stop = threading.Event()

            # 通知统计信息
            self._notification_stats = {
                'total_notifications': 0,
                'successful_notifications': 0,
                'failed_notifications': 0,
                'skipped_notifications': 0,
                'batch_notifications': 0
            }

            # 定期运行缓冲区
            self._throttled_events: Dict[
                str, Dict[str, ConfigChangeEvent]] = defaultdict(dict)

            self._initialized = True

            # 如果启用批处理，启动批处理线程
            if self._batch_mode:
                self._start_batch_processing()

    def set_categories(self, categories: Dict[str, List[str]]) -> None:
        """设置配置分类

        Args:
            categories: 分类映射，格式：{category_name: [key1, key2, ...]}
        """
        with self._lock:
            self._categories = categories.copy()

    def enable_batch_mode(self, enable: bool = True,
                          interval: Optional[float] = None) -> None:
        """启用或禁用批处理模式

        Args:
            enable: 是否启用
            interval: 批处理间隔（秒）
        """
        with self._lock:
            # 如果模式没有改变，跳过
            if self._batch_mode == enable:
                # 只更新间隔
                if interval is not None and enable:
                    self._batch_interval = interval
                return

            self._batch_mode = enable

            if interval is not None:
                self._batch_interval = interval

            if enable:
                self._start_batch_processing()
            else:
                self._stop_batch_processing()

    def _start_batch_processing(self) -> None:
        """启动批处理线程"""
        if self._batch_thread and self._batch_thread.is_alive():
            return

        self._should_stop.clear()
        self._batch_thread = threading.Thread(
            target=self._batch_processing_loop,
            daemon=True
        )
        self._batch_thread.start()
        logger.info("Started config change batch processing thread")

    def _stop_batch_processing(self) -> None:
        """停止批处理线程"""
        if not self._batch_thread or not self._batch_thread.is_alive():
            return

        self._should_stop.set()
        try:
            self._batch_thread.join(timeout=1.0)
        except:
            pass
        logger.info("Stopped config change batch processing thread")

    def _batch_processing_loop(self) -> None:
        """批处理循环"""
        batch: Dict[str, ConfigChangeEvent] = {}

        while not self._should_stop.is_set():
            try:
                # 等待事件，但定期检查停止标志
                try:
                    event = self._batch_queue.get(timeout=self._batch_interval)
                    if event.key in batch:
                        # 如果已经有相同键的事件，合并它们
                        batch[event.key] = batch[event.key].merge(event)
                    else:
                        batch[event.key] = event
                except queue.Empty:
                    pass

                # 如果有事件待处理且达到批处理间隔或队列为空
                if batch and (self._batch_queue.empty() or len(batch) >= 10):
                    self._process_batch(list(batch.values()))
                    batch.clear()

                    # 更新统计信息
                    with self._lock:
                        self._notification_stats['batch_notifications'] += 1

            except Exception as e:
                logger.error(f"Error in batch processing loop: {str(e)}")

            # 短暂休眠以避免CPU过度使用
            time.sleep(0.01)

    def _process_batch(self, events: List[ConfigChangeEvent]) -> None:
        """处理事件批次

        Args:
            events: 事件列表
        """
        if not events:
            return

        # 按优先级对订阅者排序
        subscribers = sorted(
            self._subscribers.values(),
            key=lambda s: s.priority,
            reverse=True
        )

        current_time = time.time()

        for subscriber in subscribers:
            # 检查此订阅者关心的事件
            relevant_events = []
            for event in events:
                if subscriber.matches(event.key, self._categories):
                    relevant_events.append(event)

            if not relevant_events:
                continue

            # 检查是否应该通知（限流）
            if not subscriber.should_notify(current_time):
                with self._lock:
                    self._notification_stats['skipped_notifications'] += 1
                continue

            try:
                # 对于每个相关事件通知订阅者
                for event in relevant_events:
                    subscriber.notify(event)

                    with self._lock:
                        self._notification_stats[
                            'successful_notifications'] += 1
                        self._notification_stats['total_notifications'] += 1
            except Exception as e:
                logger.error(
                    f"Error notifying subscriber {subscriber.id}: {str(e)}")

                with self._lock:
                    self._notification_stats['failed_notifications'] += 1
                    self._notification_stats['total_notifications'] += 1

    def subscribe(
            self,
            subscriber_id: str,
            callback: Callable[[ConfigChangeEvent], None],
            subscription_type: SubscriptionType = SubscriptionType.ALL,
            subscription_filter: Any = None,
            priority: int = 0,
            notification_mode: NotificationMode = NotificationMode.IMMEDIATE,
            throttle_interval: float = 0.0,
            debounce_interval: float = 0.0
    ) -> None:
        """订阅配置变更

        Args:
            subscriber_id: 订阅者ID
            callback: 回调函数
            subscription_type: 订阅类型
            subscription_filter: 订阅过滤器
            priority: 优先级
            notification_mode: 通知模式
            throttle_interval: 限流间隔（秒）
            debounce_interval: 去抖动间隔（秒）
        """
        with self._lock:
            if notification_mode != NotificationMode.IMMEDIATE and not self._batch_mode:
                # 如果需要批处理模式但尚未启用，自动启用
                self.enable_batch_mode(True)

            # 创建订阅者
            subscriber = ConfigSubscriber(
                subscriber_id=subscriber_id,
                callback=callback,
                subscription_type=subscription_type,
                subscription_filter=subscription_filter,
                priority=priority,
                throttle_interval=throttle_interval,
                debounce_interval=debounce_interval
            )

            # 注册订阅者
            self._subscribers[subscriber_id] = subscriber

            logger.debug(f"Registered config subscriber: {subscriber}")

    def subscribe_exact(
            self,
            subscriber_id: str,
            key: str,
            callback: Callable[[ConfigChangeEvent], None],
            **kwargs
    ) -> None:
        """订阅特定配置键的变更

        Args:
            subscriber_id: 订阅者ID
            key: 配置键
            callback: 回调函数
            **kwargs: 其他订阅选项
        """
        self.subscribe(
            subscriber_id=subscriber_id,
            callback=callback,
            subscription_type=SubscriptionType.EXACT,
            subscription_filter=key,
            **kwargs
        )

    def subscribe_prefix(
            self,
            subscriber_id: str,
            prefix: str,
            callback: Callable[[ConfigChangeEvent], None],
            **kwargs
    ) -> None:
        """订阅特定前缀的配置键变更

        Args:
            subscriber_id: 订阅者ID
            prefix: 配置键前缀
            callback: 回调函数
            **kwargs: 其他订阅选项
        """
        self.subscribe(
            subscriber_id=subscriber_id,
            callback=callback,
            subscription_type=SubscriptionType.PREFIX,
            subscription_filter=prefix,
            **kwargs
        )

    def subscribe_pattern(
            self,
            subscriber_id: str,
            pattern: str,
            callback: Callable[[ConfigChangeEvent], None],
            **kwargs
    ) -> None:
        """订阅匹配模式的配置键变更

        Args:
            subscriber_id: 订阅者ID
            pattern: 正则表达式模式
            callback: 回调函数
            **kwargs: 其他订阅选项
        """
        self.subscribe(
            subscriber_id=subscriber_id,
            callback=callback,
            subscription_type=SubscriptionType.PATTERN,
            subscription_filter=pattern,
            **kwargs
        )

    def subscribe_category(
            self,
            subscriber_id: str,
            category: str,
            callback: Callable[[ConfigChangeEvent], None],
            **kwargs
    ) -> None:
        """订阅特定分类的配置键变更

        Args:
            subscriber_id: 订阅者ID
            category: 分类名称
            callback: 回调函数
            **kwargs: 其他订阅选项
        """
        self.subscribe(
            subscriber_id=subscriber_id,
            callback=callback,
            subscription_type=SubscriptionType.CATEGORY,
            subscription_filter=category,
            **kwargs
        )

    def subscribe_all(
            self,
            subscriber_id: str,
            callback: Callable[[ConfigChangeEvent], None],
            **kwargs
    ) -> None:
        """订阅所有配置变更

        Args:
            subscriber_id: 订阅者ID
            callback: 回调函数
            **kwargs: 其他订阅选项
        """
        self.subscribe(
            subscriber_id=subscriber_id,
            callback=callback,
            subscription_type=SubscriptionType.ALL,
            **kwargs
        )

    def unsubscribe(self, subscriber_id: str) -> bool:
        """取消订阅配置变更

        Args:
            subscriber_id: 订阅者ID

        Returns:
            bool: 是否成功取消订阅
        """
        with self._lock:
            if subscriber_id in self._subscribers:
                # 取消任何待处理的通知
                self._subscribers[subscriber_id].cancel_pending_notifications()
                # 删除订阅者
                del self._subscribers[subscriber_id]
                logger.debug(f"Unregistered config subscriber: {subscriber_id}")
                return True
            return False

    def notify(self, key: str, old_value: Any, new_value: Any,
               source: str = None) -> None:
        """通知配置变更

        Args:
            key: 配置键
            old_value: 旧值
            new_value: 新值
            source: 通知来源
        """
        # 创建事件
        event = ConfigChangeEvent(key, old_value, new_value, source=source)

        # 如果启用了批处理模式，将事件放入队列
        if self._batch_mode:
            self._batch_queue.put(event)
            return

        # 否则，立即处理
        self._notify_subscribers(event)

    def _notify_subscribers(self, event: ConfigChangeEvent) -> None:
        """通知订阅者配置变更

        Args:
            event: 配置变更事件
        """
        # 按优先级对订阅者排序
        subscribers = sorted(
            self._subscribers.values(),
            key=lambda s: s.priority,
            reverse=True
        )

        current_time = time.time()

        for subscriber in subscribers:
            # 检查此订阅者是否关心此事件
            if not subscriber.matches(event.key, self._categories):
                continue

            # 检查是否应该通知（限流）
            if not subscriber.should_notify(current_time):
                with self._lock:
                    self._notification_stats['skipped_notifications'] += 1
                continue

            try:
                # 如果是去抖动模式，安排延迟通知
                if subscriber.debounce_interval > 0:
                    subscriber.schedule_debounced_notification(event)
                # 否则立即通知
                else:
                    subscriber.notify(event)

                with self._lock:
                    self._notification_stats['successful_notifications'] += 1
                    self._notification_stats['total_notifications'] += 1
            except Exception as e:
                logger.error(
                    f"Error notifying subscriber {subscriber.id} about {event.key}: {str(e)}")

                with self._lock:
                    self._notification_stats['failed_notifications'] += 1
                    self._notification_stats['total_notifications'] += 1

    def notify_batch(self, changes: Dict[str, Tuple[Any, Any]],
                     source: str = None) -> None:
        """批量通知配置变更

        Args:
            changes: 配置变更字典，格式：{key: (old_value, new_value)}
            source: 通知来源
        """
        # 如果启用了批处理模式，将所有事件放入队列
        if self._batch_mode:
            for key, (old_value, new_value) in changes.items():
                event = ConfigChangeEvent(key, old_value, new_value,
                                          source=source)
                self._batch_queue.put(event)
            return

        # 否则，创建事件列表立即处理
        events = [
            ConfigChangeEvent(key, old_value, new_value, source=source)
            for key, (old_value, new_value) in changes.items()
        ]

        self._process_batch(events)

    def get_stats(self) -> Dict[str, int]:
        """获取通知统计信息

        Returns:
            Dict[str, int]: 统计信息
        """
        with self._lock:
            return self._notification_stats.copy()

    def reset_stats(self) -> None:
        """重置通知统计信息"""
        with self._lock:
            for key in self._notification_stats:
                self._notification_stats[key] = 0

    def get_subscriber_count(self) -> int:
        """获取订阅者数量

        Returns:
            int: 订阅者数量
        """
        with self._lock:
            return len(self._subscribers)

    def get_subscribers(self) -> Dict[str, ConfigSubscriber]:
        """获取所有订阅者

        Returns:
            Dict[str, ConfigSubscriber]: 订阅者字典
        """
        with self._lock:
            return self._subscribers.copy()

    def shutdown(self) -> None:
        """关闭通知器，清理资源"""
        logger.info("Shutting down config change notifier")

        # 停止批处理线程
        self._stop_batch_processing()

        # 取消所有待处理的通知
        with self._lock:
            for subscriber in self._subscribers.values():
                subscriber.cancel_pending_notifications()


# 获取通知器实例的工厂函数
def get_notifier(batch_interval: float = 0.1,
                 enable_batch_mode: bool = False) -> ConfigChangeNotifier:
    """获取配置变更通知器实例

    Args:
        batch_interval: 批处理间隔（秒）
        enable_batch_mode: 是否启用批处理模式

    Returns:
        ConfigChangeNotifier: 通知器实例
    """
    return ConfigChangeNotifier(batch_interval, enable_batch_mode)


# 辅助函数：创建对配置变更的通用处理器
def create_handler(handler_name: str,
                   callback: Callable[[str, Any, Any], None]) -> Callable[
    [ConfigChangeEvent], None]:
    """创建配置变更事件处理器

    Args:
        handler_name: 处理器名称（用于日志）
        callback: 回调函数，接收(key, old_value, new_value)

    Returns:
        Callable[[ConfigChangeEvent], None]: 事件处理器
    """

    def handler(event: ConfigChangeEvent) -> None:
        try:
            callback(event.key, event.old_value, event.new_value)
        except Exception as e:
            logger.error(f"Error in config handler '{handler_name}': {str(e)}")

    return handler


# 辅助函数：自动绑定组件的配置变更处理
def bind_component_config(
        component: Any,
        component_name: str,
        config_prefix: str,
        handler_method: Optional[str] = None,
        notifier: Optional[ConfigChangeNotifier] = None
) -> None:
    """绑定组件的配置变更处理

    Args:
        component: 组件实例
        component_name: 组件名称
        config_prefix: 配置前缀
        handler_method: 处理方法名，如果为None，则尝试使用on_config_changed
        notifier: 通知器实例，如果为None，则使用默认实例
    """
    # 获取通知器
    if notifier is None:
        notifier = get_notifier()

    # 确定处理方法
    method_name = handler_method or 'on_config_changed'
    if not hasattr(component, method_name) or not callable(
            getattr(component, method_name)):
        logger.warning(
            f"Component {component_name} does not have method {method_name}")
        return

    handler = getattr(component, method_name)

    # 创建订阅者ID
    subscriber_id = f"{component_name}_{config_prefix}"

    # 订阅前缀匹配的配置变更
    notifier.subscribe_prefix(
        subscriber_id=subscriber_id,
        prefix=config_prefix,
        callback=create_handler(subscriber_id, handler)
    )

    logger.debug(
        f"Bound component {component_name} to config prefix {config_prefix}")


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 获取通知器实例
    notifier = get_notifier(enable_batch_mode=True)

    # 设置配置分类
    categories = {
        "ui": ["ui.show_debug_info", "ui.theme", "ui.camera_width",
               "ui.camera_height"],
        "detector": ["detector.use_mediapipe", "detector.performance_mode",
                     "detector.confidence_threshold"]
    }
    notifier.set_categories(categories)


    # 定义一些处理函数
    def handle_detector_config(key: str, old_value: Any,
                               new_value: Any) -> None:
        print(
            f"[Detector] Config changed: {key} = {new_value} (was: {old_value})")


    def handle_ui_config(key: str, old_value: Any, new_value: Any) -> None:
        print(f"[UI] Config changed: {key} = {new_value} (was: {old_value})")


    def handle_performance_mode(key: str, old_value: Any,
                                new_value: Any) -> None:
        print(f"[Performance] Mode changed to {new_value} (was: {old_value})")


    def handle_all_changes(key: str, old_value: Any, new_value: Any) -> None:
        print(f"[All] Config changed: {key} = {new_value} (was: {old_value})")


    # 订阅变更
    notifier.subscribe_category("ui_handler", "ui",
                                create_handler("ui_handler", handle_ui_config))
    notifier.subscribe_category("detector_handler", "detector",
                                create_handler("detector_handler",
                                               handle_detector_config))
    notifier.subscribe_exact("performance_handler", "detector.performance_mode",
                             create_handler("performance_handler",
                                            handle_performance_mode))
    notifier.subscribe_all("all_handler",
                           create_handler("all_handler", handle_all_changes),
                           priority=-10)  # 低优先级

    # 测试通知
    print("Testing individual notifications:")
    notifier.notify("ui.theme", "light", "dark")
    notifier.notify("detector.performance_mode", "balanced", "accurate")

    # 等待批处理完成
    time.sleep(0.2)

    # 测试批量通知
    print("\nTesting batch notifications:")
    batch_changes = {
        "ui.camera_width": (640, 800),
        "ui.camera_height": (480, 600),
        "detector.use_mediapipe": (False, True)
    }
    notifier.notify_batch(batch_changes)

    # 等待批处理完成
    time.sleep(0.2)

    # 获取统计信息
    stats = notifier.get_stats()
    print("\nNotification stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 关闭通知器
    notifier.shutdown()
