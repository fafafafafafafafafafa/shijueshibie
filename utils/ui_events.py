# -*- coding: utf-8 -*-
"""
第1部分：事件定义和核心数据结构

UI事件系统 - 提供UI组件与事件系统的集成

本模块提供:
1. 专用的UI事件类型定义
2. UI组件与事件系统的集成接口
3. 事件驱动的UI更新机制
4. 事件缓存和对象池机制
5. 异步和批处理支持
6. 性能监控和统计功能
7. 事件记录和回放
"""
from typing import Dict, List, Any, Callable, Optional, Union, Tuple, Set
from enum import Enum, auto
import time
import threading
import queue
import weakref
import uuid
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

from utils.event_system import get_event_system
from utils.conditional_events import get_conditional_event_system, where, \
    Condition
from utils.logger_config import get_logger

# 获取日志记录器
logger = get_logger("UIEvents")


class UIEventTypes:
    """UI事件类型常量"""
    # 窗口事件
    WINDOW_CREATED = "window_created"
    WINDOW_RESIZED = "window_resized"
    WINDOW_CLOSED = "window_closed"
    WINDOW_FOCUSED = "window_focused"
    WINDOW_BLURRED = "window_blurred"
    WINDOW_MOVED = "window_moved"
    WINDOW_MINIMIZED = "window_minimized"
    WINDOW_MAXIMIZED = "window_maximized"
    WINDOW_RESTORED = "window_restored"

    # 显示事件
    DISPLAY_UPDATED = "display_updated"
    FRAME_RENDERED = "frame_rendered"
    RENDER_COMPLETED = "render_completed"
    RENDER_STARTED = "render_started"
    BUFFER_SWAPPED = "buffer_swapped"
    SCREEN_REFRESH = "screen_refresh"
    VSYNC_EVENT = "vsync_event"

    # 用户交互事件
    KEY_PRESSED = "key_pressed"
    KEY_RELEASED = "key_released"
    KEY_TYPED = "key_typed"
    MOUSE_CLICKED = "mouse_clicked"
    MOUSE_PRESSED = "mouse_pressed"
    MOUSE_RELEASED = "mouse_released"
    MOUSE_MOVED = "mouse_moved"
    MOUSE_DRAGGED = "mouse_dragged"
    MOUSE_ENTERED = "mouse_entered"
    MOUSE_EXITED = "mouse_exited"
    MOUSE_WHEEL = "mouse_wheel"
    TOUCH_START = "touch_start"
    TOUCH_MOVE = "touch_move"
    TOUCH_END = "touch_end"
    GESTURE_START = "gesture_start"
    GESTURE_UPDATE = "gesture_update"
    GESTURE_END = "gesture_end"

    # 控件事件
    BUTTON_CLICKED = "button_clicked"
    MENU_SELECTED = "menu_selected"
    LIST_ITEM_SELECTED = "list_item_selected"
    TEXTBOX_CHANGED = "textbox_changed"
    SLIDER_CHANGED = "slider_changed"
    CHECKBOX_CHANGED = "checkbox_changed"
    DIALOG_OPENED = "dialog_opened"
    DIALOG_CLOSED = "dialog_closed"
    POPUP_SHOWN = "popup_shown"
    POPUP_HIDDEN = "popup_hidden"
    TOOLTIP_SHOWN = "tooltip_shown"
    TOOLTIP_HIDDEN = "tooltip_hidden"

    # UI状态事件
    VIEW_MODE_CHANGED = "view_mode_changed"
    OPTION_TOGGLED = "option_toggled"
    FEATURE_TOGGLED = "feature_toggled"
    THEME_CHANGED = "theme_changed"
    LANGUAGE_CHANGED = "language_changed"
    LAYOUT_CHANGED = "layout_changed"
    ZOOM_CHANGED = "zoom_changed"
    NAVIGATION_CHANGED = "navigation_changed"

    # 信息展示事件
    DEBUG_INFO_UPDATED = "debug_info_updated"
    STATUS_CHANGED = "status_changed"
    NOTIFICATION_SHOWN = "notification_shown"
    NOTIFICATION_CLICKED = "notification_clicked"
    NOTIFICATION_CLOSED = "notification_closed"
    PROGRESS_UPDATED = "progress_updated"
    ALERT_SHOWN = "alert_shown"

    # 性能事件
    FPS_UPDATED = "fps_updated"
    PERFORMANCE_WARNING = "performance_warning"
    CPU_USAGE_UPDATED = "cpu_usage_updated"
    MEMORY_USAGE_UPDATED = "memory_usage_updated"
    BATTERY_STATUS_UPDATED = "battery_status_updated"
    NETWORK_STATUS_UPDATED = "network_status_updated"
    LAG_DETECTED = "lag_detected"

    # 数据事件
    DATA_LOADED = "data_loaded"
    DATA_SAVED = "data_saved"
    DATA_UPDATED = "data_updated"
    DATA_PROCESSING_STARTED = "data_processing_started"
    DATA_PROCESSING_COMPLETED = "data_processing_completed"
    DATA_PROCESSING_ERROR = "data_processing_error"

    # 系统事件
    SYSTEM_INFO_UPDATED = "system_info_updated"
    APP_STATE_CHANGED = "app_state_changed"
    LOW_MEMORY_WARNING = "low_memory_warning"
    ERROR_OCCURRED = "error_occurred"
    LOG_ADDED = "log_added"
    HEARTBEAT = "heartbeat"


class Priority(Enum):
    """事件处理优先级"""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 0


class BatchingPolicy(Enum):
    """事件批处理策略"""
    NONE = auto()  # 不进行批处理
    COALESCE = auto()  # 合并相同事件类型的最新值
    THROTTLE = auto()  # 按时间间隔限制事件频率
    DEBOUNCE = auto()  # 等待一段时间无事件后再处理


class EventCacheEntry:
    """事件缓存条目"""

    def __init__(self, event_type: str, data: Dict[str, Any],
                 timestamp: float = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self.processed = False
        self.batch_id = None
        self.event_id = str(uuid.uuid4())


class EventPerformanceStats:
    """事件性能统计"""

    def __init__(self):
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.processing_times: Dict[str, List[float]] = defaultdict(list)
        self.handler_times: Dict[
            str, Dict[Callable, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        self.max_processing_times: Dict[str, float] = defaultdict(float)
        self.slow_handlers: Set[Tuple[str, Callable]] = set()
        self.samples_limit = 100

    def record_event(self, event_type: str) -> None:
        """记录事件发生"""
        self.event_counts[event_type] += 1

    def record_processing_time(self, event_type: str, elapsed: float) -> None:
        """记录事件处理总时间"""
        self.processing_times[event_type].append(elapsed)
        if len(self.processing_times[event_type]) > self.samples_limit:
            self.processing_times[event_type].pop(0)

        if elapsed > self.max_processing_times[event_type]:
            self.max_processing_times[event_type] = elapsed

    def record_handler_time(self, event_type: str, handler: Callable,
                            elapsed: float) -> None:
        """记录特定处理器的处理时间"""
        self.handler_times[event_type][handler].append(elapsed)
        if len(self.handler_times[event_type][handler]) > self.samples_limit:
            self.handler_times[event_type][handler].pop(0)

        # 标记慢处理器 (> 16ms, 约为60fps的一帧时间)
        if elapsed > 0.016:
            self.slow_handlers.add((event_type, handler))

    def get_average_processing_time(self, event_type: str) -> float:
        """获取事件平均处理时间"""
        times = self.processing_times.get(event_type, [])
        return sum(times) / len(times) if times else 0

    def get_average_handler_time(self, event_type: str,
                                 handler: Callable) -> float:
        """获取处理器平均执行时间"""
        times = self.handler_times.get(event_type, {}).get(handler, [])
        return sum(times) / len(times) if times else 0

    def get_slow_handlers(self) -> List[Tuple[str, Callable, float]]:
        """获取慢处理器列表"""
        result = []
        for event_type, handler in self.slow_handlers:
            avg_time = self.get_average_handler_time(event_type, handler)
            result.append((event_type, handler, avg_time))
        return sorted(result, key=lambda x: x[2], reverse=True)

    def get_summary(self) -> Dict[str, Any]:
        """获取性能统计摘要"""
        return {
            "total_events": sum(self.event_counts.values()),
            "events_by_type": dict(self.event_counts),
            "average_times": {
                event_type: self.get_average_processing_time(event_type)
                for event_type in self.processing_times
            },
            "max_times": dict(self.max_processing_times),
            "slow_handlers_count": len(self.slow_handlers)
        }

# 第1部分结束：事件定义和核心数据结构
# -*- coding: utf-8 -*-
"""
第2部分：事件发布器类
"""


class UIEventPublisher:
    """
    UI事件发布器 - 提供UI组件发布事件的统一接口

    此类作为UI组件和事件系统之间的桥梁，提供友好的API来发布UI相关事件。
    支持事件缓存、批处理和异步发布。
    """
    # 实例缓存，避免重复创建发布器实例
    _instances = {}

    @classmethod
    def get_instance(cls, component_name=None) -> 'UIEventPublisher':
        """获取或创建发布器实例"""
        component_name = component_name or "unknown_component"
        if component_name not in cls._instances:
            cls._instances[component_name] = cls(component_name)
        return cls._instances[component_name]

    def __init__(self, component_name=None):
        """
        初始化UI事件发布器

        Args:
            component_name: UI组件名称，用于标识事件源
        """
        self.component_name = component_name or "unknown_component"
        self.event_system = get_event_system()
        self.batching_enabled = False
        self.batching_interval = 0.05  # 50ms批处理间隔
        self.batching_policies = {}  # 按事件类型存储批处理策略
        self.event_cache = {}  # 事件缓存
        self.batch_timer = None
        self.batch_lock = threading.RLock()
        self.async_executor = ThreadPoolExecutor(max_workers=2,
                                                 thread_name_prefix=f"{component_name}_publisher")
        self.performance_stats = EventPerformanceStats()
        self._object_pool = deque(maxlen=100)  # 对象池，重用事件数据字典

        logger.info(f"UI事件发布器已初始化: {self.component_name}")

    def _get_data_dict(self) -> Dict[str, Any]:
        """从对象池获取数据字典或创建新的"""
        if self._object_pool:
            data = self._object_pool.pop()
            data.clear()
        else:
            data = {}

        # 添加基础字段
        data["component"] = self.component_name
        data["timestamp"] = time.time()

        return data

    def _recycle_data_dict(self, data: Dict[str, Any]) -> None:
        """回收数据字典到对象池"""
        if len(self._object_pool) < self._object_pool.maxlen:
            self._object_pool.append(data)

    def enable_batching(self, interval: float = 0.05) -> None:
        """
        启用事件批处理

        Args:
            interval: 批处理间隔时间（秒）
        """
        with self.batch_lock:
            self.batching_enabled = True
            self.batching_interval = interval
            logger.debug(
                f"UI组件 {self.component_name} 已启用事件批处理, 间隔: {interval}秒")

    def disable_batching(self) -> None:
        """禁用事件批处理并立即处理所有挂起的事件"""
        with self.batch_lock:
            self.batching_enabled = False
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None

            # 处理挂起的事件
            self._process_batched_events()
            logger.debug(f"UI组件 {self.component_name} 已禁用事件批处理")

    def set_batching_policy(self, event_type: str,
                            policy: BatchingPolicy) -> None:
        """
        为特定事件类型设置批处理策略

        Args:
            event_type: 事件类型
            policy: 批处理策略
        """
        self.batching_policies[event_type] = policy
        logger.debug(f"为事件类型 {event_type} 设置批处理策略: {policy.name}")

    def _schedule_batch_processing(self) -> None:
        """调度批处理任务"""
        with self.batch_lock:
            if self.batch_timer:
                # 已经有计时器在运行
                return

            def process_batch():
                with self.batch_lock:
                    self.batch_timer = None
                    self._process_batched_events()

            self.batch_timer = threading.Timer(self.batching_interval,
                                               process_batch)
            self.batch_timer.daemon = True
            self.batch_timer.start()

    def _process_batched_events(self) -> None:
        """处理所有批处理事件"""
        with self.batch_lock:
            if not self.event_cache:
                return

            # 按事件类型分组处理
            for event_type, events in list(self.event_cache.items()):
                policy = self.batching_policies.get(event_type,
                                                    BatchingPolicy.NONE)

                if policy == BatchingPolicy.COALESCE:
                    # 只发布最新的事件
                    latest_event = max(events, key=lambda e: e.timestamp)
                    self.event_system.publish(event_type, latest_event.data)
                    self.performance_stats.record_event(event_type)

                elif policy == BatchingPolicy.THROTTLE:
                    # 发布第一个和最后一个事件
                    first_event = min(events, key=lambda e: e.timestamp)
                    last_event = max(events, key=lambda e: e.timestamp)

                    self.event_system.publish(event_type, first_event.data)

                    if first_event != last_event:
                        self.event_system.publish(event_type, last_event.data)

                    self.performance_stats.record_event(event_type)

                else:
                    # 默认行为：发布所有事件
                    for event in events:
                        if not event.processed:
                            self.event_system.publish(event_type, event.data)
                            event.processed = True
                            self.performance_stats.record_event(event_type)

                # 清理并回收对象
                for event in events:
                    self._recycle_data_dict(event.data)

                # 从缓存中移除
                del self.event_cache[event_type]

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        发布UI事件

        Args:
            event_type: 事件类型
            data: 事件数据
        """
        # 记录性能数据
        self.performance_stats.record_event(event_type)

        if not self.batching_enabled or event_type not in self.batching_policies:
            # 直接发布
            self.event_system.publish(event_type, data)
            return

        # 批处理逻辑
        with self.batch_lock:
            if event_type not in self.event_cache:
                self.event_cache[event_type] = []

            # 创建缓存条目
            entry = EventCacheEntry(event_type, data)
            self.event_cache[event_type].append(entry)

            # 调度处理
            self._schedule_batch_processing()

    def publish_async(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        异步发布UI事件

        Args:
            event_type: 事件类型
            data: 事件数据
        """
        self.async_executor.submit(self.publish, event_type, data)

    def publish_window_event(self, event_type: str, window_name: str,
                             window_size=None,
                             **kwargs) -> None:
        """
        发布窗口相关事件

        Args:
            event_type: 事件类型，如 UIEventTypes.WINDOW_CREATED
            window_name: 窗口名称
            window_size: 窗口尺寸 (width, height)
            **kwargs: 其他相关数据
        """
        data = self._get_data_dict()
        data.update({
            "window_name": window_name,
            **kwargs
        })

        if window_size:
            data["window_size"] = window_size

        self.publish(event_type, data)

    def publish_display_event(self, event_type: str, display_data=None,
                              frame_info=None, **kwargs) -> None:
        """
        发布显示相关事件

        Args:
            event_type: 事件类型，如 UIEventTypes.DISPLAY_UPDATED
            display_data: 显示相关数据
            frame_info: 帧相关信息，如尺寸、类型等
            **kwargs: 其他相关数据
        """
        data = self._get_data_dict()
        data.update(kwargs)

        if display_data:
            data["display_data"] = display_data

        if frame_info:
            data["frame_info"] = frame_info

        self.publish(event_type, data)

    def publish_user_interaction(self, event_type: str, interaction_type: str,
                                 value: Any,
                                 position=None, **kwargs) -> None:
        """
        发布用户交互事件

        Args:
            event_type: 事件类型，如 UIEventTypes.KEY_PRESSED
            interaction_type: 交互类型，如 "keyboard", "mouse"
            value: 交互值，如按键代码、鼠标位置等
            position: 交互位置，如鼠标坐标
            **kwargs: 其他相关数据
        """
        data = self._get_data_dict()
        data.update({
            "interaction_type": interaction_type,
            "value": value,
            **kwargs
        })

        if position:
            data["position"] = position

        self.publish(event_type, data)

    def publish_ui_state_change(self, event_type: str, state_name: str,
                                old_value: Any,
                                new_value: Any, **kwargs) -> None:
        """
        发布UI状态变更事件

        Args:
            event_type: 事件类型，如 UIEventTypes.OPTION_TOGGLED
            state_name: 状态名称
            old_value: 旧值
            new_value: 新值
            **kwargs: 其他相关数据
        """
        data = self._get_data_dict()
        data.update({
            "state_name": state_name,
            "old_value": old_value,
            "new_value": new_value,
            **kwargs
        })

        self.publish(event_type, data)

    def publish_notification(self, message: str, level: str = "info",
                             duration: float = 3.0,
                             action=None, **kwargs) -> str:
        """
        发布通知事件

        Args:
            message: 通知消息
            level: 通知级别 (info, warning, error)
            duration: 通知显示时长(秒)
            action: 可选的通知动作
            **kwargs: 其他相关数据

        Returns:
            str: 通知ID
        """
        notification_id = str(uuid.uuid4())

        data = self._get_data_dict()
        data.update({
            "message": message,
            "level": level,
            "duration": duration,
            "notification_id": notification_id,
            **kwargs
        })

        if action:
            data["action"] = action

        self.publish(UIEventTypes.NOTIFICATION_SHOWN, data)
        return notification_id

    def publish_fps_update(self, fps: float, frame_time=None, **kwargs) -> None:
        """
        发布FPS更新事件

        Args:
            fps: 当前帧率
            frame_time: 单帧处理时间(毫秒)
            **kwargs: 其他相关数据
        """
        data = self._get_data_dict()
        data.update({
            "fps": fps,
            **kwargs
        })

        if frame_time:
            data["frame_time"] = frame_time

        self.publish(UIEventTypes.FPS_UPDATED, data)

        # 如果FPS过低，也发布性能警告事件
        if fps < 15:
            warning_data = self._get_data_dict()
            warning_data.update({
                "warning_type": "low_fps",
                "fps": fps,
                **kwargs
            })
            self.publish(UIEventTypes.PERFORMANCE_WARNING, warning_data)

    def publish_performance_warning(self, warning_type: str,
                                    details: Dict[str, Any]) -> None:
        """
        发布性能警告事件

        Args:
            warning_type: 警告类型
            details: 警告详细信息
        """
        data = self._get_data_dict()
        data.update({
            "warning_type": warning_type,
            "details": details
        })

        self.publish(UIEventTypes.PERFORMANCE_WARNING, data)

    def publish_control_event(self, control_type: str, control_id: str,
                              value: Any, event_type: str = None,
                              **kwargs) -> None:
        """
        发布控件事件

        Args:
            control_type: 控件类型 (button, slider等)
            control_id: 控件ID
            value: 控件值
            event_type: 事件类型，如自定义或预定义类型
            **kwargs: 其他相关数据
        """
        if event_type is None:
            # 根据控件类型选择事件
            if control_type == "button":
                event_type = UIEventTypes.BUTTON_CLICKED
            elif control_type == "list":
                event_type = UIEventTypes.LIST_ITEM_SELECTED
            elif control_type == "textbox":
                event_type = UIEventTypes.TEXTBOX_CHANGED
            elif control_type == "slider":
                event_type = UIEventTypes.SLIDER_CHANGED
            elif control_type == "checkbox":
                event_type = UIEventTypes.CHECKBOX_CHANGED
            else:
                # 默认使用状态改变事件
                event_type = UIEventTypes.OPTION_TOGGLED

        data = self._get_data_dict()
        data.update({
            "control_type": control_type,
            "control_id": control_id,
            "value": value,
            **kwargs
        })

        self.publish(event_type, data)

    def publish_debug_info(self, category: str, info: Dict[str, Any]) -> None:
        """
        发布调试信息

        Args:
            category: 信息类别
            info: 调试信息
        """
        data = self._get_data_dict()
        data.update({
            "category": category,
            "info": info
        })

        self.publish(UIEventTypes.DEBUG_INFO_UPDATED, data)

    def publish_system_event(self, event_type: str,
                             system_data: Dict[str, Any]) -> None:
        """
        发布系统事件

        Args:
            event_type: 事件类型
            system_data: 系统数据
        """
        data = self._get_data_dict()
        data.update(system_data)

        self.publish(event_type, data)

    def publish_heartbeat(self, status: str = "alive",
                          details: Dict[str, Any] = None) -> None:
        """
        发布心跳事件，用于监控组件活动状态

        Args:
            status: 状态
            details: 详细信息
        """
        data = self._get_data_dict()
        data.update({
            "status": status
        })

        if details:
            data["details"] = details

        self.publish(UIEventTypes.HEARTBEAT, data)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取发布器性能统计"""
        return self.performance_stats.get_summary()

    def cleanup(self) -> None:
        """清理资源"""
        self.disable_batching()
        self.async_executor.shutdown(wait=False)
        self.event_cache.clear()
        self._object_pool.clear()

# 第2部分结束：事件发布器类
# -*- coding: utf-8 -*-
"""
第3部分：事件订阅器类 (第一部分)
"""
# -*- coding: utf-8 -*-
"""
第3部分和第4部分：事件订阅器类
"""


class UIEventSubscriber:
    """
    UI事件订阅器 - 提供UI组件订阅事件的统一接口

    此类作为UI组件和事件系统之间的桥梁，提供友好的API来订阅和处理UI相关事件。
    支持异步处理、性能监控和自动资源管理。
    """
    # 实例缓存，避免重复创建订阅器实例
    _instances = {}

    @classmethod
    def get_instance(cls, component_name=None) -> 'UIEventSubscriber':
        """获取或创建订阅器实例"""
        component_name = component_name or "unknown_component"
        if component_name not in cls._instances:
            cls._instances[component_name] = cls(component_name)
        return cls._instances[component_name]

    def __init__(self, component_name=None):
        """
        初始化UI事件订阅器

        Args:
            component_name: UI组件名称，用于标识事件处理器
        """
        self.component_name = component_name or "unknown_component"
        self.event_system = get_event_system()
        self.conditional_events = get_conditional_event_system(
            self.event_system)
        self.subscriptions = {}  # 记录已订阅的处理器
        self.async_enabled = False
        self.async_executor = None
        self.async_queue = queue.Queue()
        self.async_thread = None
        self.async_running = False
        self.performance_stats = EventPerformanceStats()
        self.async_handlers = set()  # 记录异步处理器
        self.once_handlers = set()  # 记录一次性处理器

        logger.info(f"UI事件订阅器已初始化: {self.component_name}")

    def enable_async_processing(self, max_workers: int = 2) -> None:
        """
        启用异步事件处理

        Args:
            max_workers: 最大工作线程数
        """
        if self.async_enabled:
            return

        self.async_enabled = True
        self.async_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"{self.component_name}_subscriber"
        )
        self.async_running = True

        # 启动异步处理线程
        self.async_thread = threading.Thread(
            target=self._async_processing_loop,
            name=f"{self.component_name}_event_processor",
            daemon=True
        )
        self.async_thread.start()

        logger.info(f"UI组件 {self.component_name} 已启用异步事件处理")

    def disable_async_processing(self) -> None:
        """禁用异步事件处理"""
        if not self.async_enabled:
            return

        self.async_running = False
        self.async_enabled = False

        if self.async_thread:
            # 放入终止信号
            self.async_queue.put(None)
            self.async_thread.join(timeout=1.0)
            self.async_thread = None

        if self.async_executor:
            self.async_executor.shutdown(wait=False)
            self.async_executor = None

        logger.info(f"UI组件 {self.component_name} 已禁用异步事件处理")

    def _async_processing_loop(self) -> None:
        """异步事件处理循环"""
        while self.async_running:
            try:
                # 从队列中获取事件处理任务
                task = self.async_queue.get(timeout=0.5)

                if task is None:
                    # 收到终止信号
                    break

                event_type, data, handler = task

                # 提交到线程池处理
                self.async_executor.submit(self._execute_handler, event_type,
                                           data, handler)

            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"异步事件处理发生错误: {e}")

    def _execute_handler(self, event_type: str, data: Dict[str, Any],
                         handler: Callable) -> None:
        """执行事件处理器并记录性能数据"""
        try:
            start_time = time.time()
            handler(data)
            elapsed = time.time() - start_time

            # 记录性能数据
            self.performance_stats.record_handler_time(event_type, handler,
                                                       elapsed)

            # 检测慢处理器
            if elapsed > 0.1:  # 100ms
                logger.warning(
                    f"慢事件处理器: {handler.__name__} 处理 {event_type} 用时 {elapsed:.3f}秒")

        except Exception as e:
            logger.error(
                f"事件处理器 {handler.__name__} 处理 {event_type} 时发生错误: {e}")

    def _event_handler_wrapper(self, event_type: str, handler: Callable,
                               async_mode: bool,
                               once: bool = False) -> Callable:
        """包装事件处理器，添加性能监控和异步处理"""

        def wrapped_handler(data):
            # 检查是否为一次性处理器
            if once:
                # 先取消订阅再处理，防止在处理期间又收到事件
                self.event_system.unsubscribe(event_type, wrapped_handler)
                self.once_handlers.discard(wrapped_handler)
                if event_type in self.subscriptions and wrapped_handler in \
                        self.subscriptions[event_type]:
                    self.subscriptions[event_type].remove(wrapped_handler)

            if async_mode and self.async_enabled:
                # 通过队列提交异步处理
                self.async_queue.put((event_type, data, handler))
            else:
                # 同步处理
                self._execute_handler(event_type, data, handler)

        # 保留原处理器的名称和文档
        wrapped_handler.__name__ = getattr(handler, '__name__',
                                           'unknown_handler')
        wrapped_handler.__doc__ = getattr(handler, '__doc__', None)

        # 标记处理器类型
        if once:
            self.once_handlers.add(wrapped_handler)

        if async_mode:
            self.async_handlers.add(wrapped_handler)

        return wrapped_handler

    def subscribe(self, event_type: str, handler: Callable,
                  priority: Union[int, Priority] = Priority.NORMAL,
                  async_mode: bool = False, once: bool = False) -> bool:
        """
        订阅UI事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数
            priority: 处理优先级
            async_mode: 是否异步处理
            once: 是否只处理一次

        Returns:
            bool: 是否成功订阅
        """
        try:
            # 如果使用异步模式，确保已启用异步处理
            if async_mode and not self.async_enabled:
                self.enable_async_processing()

            # 获取优先级数值
            if isinstance(priority, Priority):
                priority_value = priority.value
            else:
                priority_value = priority

            # 包装处理器
            wrapped_handler = self._event_handler_wrapper(event_type, handler,
                                                          async_mode, once)

            # 订阅事件
            success = self.event_system.subscribe_with_priority(
                event_type, wrapped_handler, priority_value)

            if success:
                # 记录订阅，用于后续取消订阅
                if event_type not in self.subscriptions:
                    self.subscriptions[event_type] = []
                self.subscriptions[event_type].append(wrapped_handler)

                logger.debug(
                    f"UI组件 {self.component_name} 订阅了事件: {event_type}"
                    f"{' [异步]' if async_mode else ''}"
                    f"{' [一次性]' if once else ''}"
                )

            return success
        except Exception as e:
            logger.error(
                f"UI组件 {self.component_name} 订阅事件 {event_type} 失败: {e}")
            return False

    def subscribe_once(self, event_type: str, handler: Callable,
                       priority: Union[int, Priority] = Priority.NORMAL,
                       async_mode: bool = False) -> bool:
        """
        订阅UI事件，只处理一次

        Args:
            event_type: 事件类型
            handler: 事件处理函数
            priority: 处理优先级
            async_mode: 是否异步处理

        Returns:
            bool: 是否成功订阅
        """
        return self.subscribe(event_type, handler, priority, async_mode,
                              once=True)

    def subscribe_async(self, event_type: str, handler: Callable,
                        priority: Union[
                            int, Priority] = Priority.NORMAL) -> bool:
        """
        异步订阅UI事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数
            priority: 处理优先级

        Returns:
            bool: 是否成功订阅
        """
        return self.subscribe(event_type, handler, priority, async_mode=True)

    def subscribe_with_timeout(self, event_type: str, handler: Callable,
                               timeout: float,
                               priority: Union[
                                   int, Priority] = Priority.NORMAL) -> None:
        """
        订阅有超时的事件，若超时未触发则调用超时处理函数

        Args:
            event_type: 事件类型
            handler: 事件处理函数
            timeout: 超时时间(秒)
            priority: 处理优先级
        """
        timeout_event = threading.Event()

        def timeout_handler():
            if not timeout_event.wait(timeout):
                # 超时未触发
                handler({"timeout": True, "event_type": event_type})

        def wrapped_handler(data):
            # 事件已触发
            timeout_event.set()
            handler(data)

        # 启动超时线程
        threading.Thread(
            target=timeout_handler,
            daemon=True,
            name=f"{self.component_name}_timeout_{event_type}"
        ).start()

        # 订阅事件
        self.subscribe_once(event_type, wrapped_handler, priority)

    def subscribe_conditional(self, event_type: str,
                              condition_builder: Union[Callable, Condition],
                              handler: Callable,
                              priority: Union[int, Priority] = Priority.NORMAL,
                              async_mode: bool = False) -> bool:
        """
        条件性订阅UI事件

        Args:
            event_type: 事件类型
            condition_builder: 条件构建器函数或条件对象
            handler: 事件处理函数
            priority: 处理优先级
            async_mode: 是否异步处理

        Returns:
            bool: 是否成功订阅
        """
        try:
            # 如果使用异步模式，确保已启用异步处理
            if async_mode and not self.async_enabled:
                self.enable_async_processing()

            # 获取优先级数值
            if isinstance(priority, Priority):
                priority_value = priority.value
            else:
                priority_value = priority

            # 包装处理器
            wrapped_handler = self._event_handler_wrapper(event_type, handler,
                                                          async_mode)

            # 订阅条件事件
            success = self.conditional_events.subscribe_if(
                event_type, condition_builder, wrapped_handler, priority_value)

            if success:
                # 记录订阅，用于后续取消订阅
                if event_type not in self.subscriptions:
                    self.subscriptions[event_type] = []
                self.subscriptions[event_type].append(wrapped_handler)

                logger.debug(
                    f"UI组件 {self.component_name} 条件性订阅了事件: {event_type}"
                    f"{' [异步]' if async_mode else ''}"
                )

            return success
        except Exception as e:
            logger.error(
                f"UI组件 {self.component_name} 条件性订阅事件 {event_type} 失败: {e}")
            return False

    def subscribe_to_notifications(self, handler: Callable, level: str = None,
                                   component: str = None, priority: Union[
                int, Priority] = Priority.NORMAL,
                                   async_mode: bool = False) -> bool:
        """
        订阅通知事件

        Args:
            handler: 通知处理函数
            level: 可选的通知级别过滤 (info, warning, error)
            component: 可选的组件名称过滤
            priority: 处理优先级
            async_mode: 是否异步处理

        Returns:
            bool: 是否成功订阅
        """
        try:
            # 创建条件构建器
            condition = Condition()

            # 添加级别条件（如果指定）
            if level:
                condition.equals('level', level)

            # 添加组件条件（如果指定）
            if component:
                condition.equals('component', component)

            # 订阅通知事件
            return self.subscribe_conditional(
                UIEventTypes.NOTIFICATION_SHOWN,
                condition,
                handler,
                priority,
                async_mode
            )
        except Exception as e:
            logger.error(f"UI组件 {self.component_name} 订阅通知事件失败: {e}")
            return False

    def subscribe_to_performance_warnings(self, handler: Callable,
                                          warning_type: str = None,
                                          priority: Union[
                                              int, Priority] = Priority.NORMAL) -> bool:
        """
        订阅性能警告事件

        Args:
            handler: 性能警告处理函数
            warning_type: 可选的警告类型过滤
            priority: 处理优先级

        Returns:
            bool: 是否成功订阅
        """
        try:
            # 创建条件构建器
            condition = Condition()

            # 添加警告类型条件（如果指定）
            if warning_type:
                condition.equals('warning_type', warning_type)

            # 订阅性能警告事件
            return self.subscribe_conditional(
                UIEventTypes.PERFORMANCE_WARNING,
                condition,
                handler,
                priority
            )
        except Exception as e:
            logger.error(
                f"UI组件 {self.component_name} 订阅性能警告事件失败: {e}")
            return False

    def unsubscribe_all(self) -> int:
        """
        取消所有已订阅的事件

        Returns:
            int: 取消的订阅数量
        """
        count = 0

        for event_type, handlers in self.subscriptions.items():
            for handler in handlers:
                try:
                    if self.event_system.unsubscribe(event_type, handler):
                        count += 1
                except Exception as e:
                    logger.error(f"取消订阅事件 {event_type} 失败: {e}")

        # 清空记录
        self.subscriptions.clear()
        self.async_handlers.clear()
        self.once_handlers.clear()

        logger.info(f"UI组件 {self.component_name} 取消了 {count} 个事件订阅")
        return count

    def unsubscribe(self, event_type: str, handler: Callable = None) -> int:
        """
        取消特定事件类型的订阅

        Args:
            event_type: 事件类型
            handler: 特定的处理函数，None表示取消该事件类型的所有订阅

        Returns:
            int: 取消的订阅数量
        """
        if event_type not in self.subscriptions:
            return 0

        count = 0

        if handler:
            # 查找包装后的处理器
            wrapped_handlers = []
            for wrapped in self.subscriptions[event_type]:
                if getattr(wrapped, '__name__', '') == getattr(handler,
                                                               '__name__', ''):
                    wrapped_handlers.append(wrapped)

            if not wrapped_handlers:
                return 0

            # 取消特定处理函数
            for wrapped_handler in wrapped_handlers:
                try:
                    if self.event_system.unsubscribe(event_type,
                                                     wrapped_handler):
                        self.subscriptions[event_type].remove(wrapped_handler)
                        self.async_handlers.discard(wrapped_handler)
                        self.once_handlers.discard(wrapped_handler)
                        count += 1
                except Exception as e:
                    logger.error(
                        f"取消订阅事件 {event_type} 的特定处理函数失败: {e}")
        else:
            # 取消该事件类型的所有订阅
            handlers = list(self.subscriptions[event_type])
            for h in handlers:
                try:
                    if self.event_system.unsubscribe(event_type, h):
                        count += 1
                        self.async_handlers.discard(h)
                        self.once_handlers.discard(h)
                except Exception as e:
                    logger.error(
                        f"取消订阅事件 {event_type} 的处理函数失败: {e}")

            # 清空该事件类型的记录
            self.subscriptions[event_type] = []

        logger.info(
            f"UI组件 {self.component_name} 取消了 {count} 个 {event_type} 事件订阅")
        return count

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取订阅器性能统计"""
        return self.performance_stats.get_summary()

    def get_slow_handlers(self) -> List[Tuple[str, Callable, float]]:
        """获取慢处理器列表"""
        return self.performance_stats.get_slow_handlers()

    def cleanup(self) -> None:
        """清理资源"""
        self.unsubscribe_all()
        self.disable_async_processing()

# 第3部分和第4部分结束：事件订阅器类
# -*- coding: utf-8 -*-
"""
第5部分：事件驱动的UI组件基类
"""


class UIEventDrivenComponent:
    """
    事件驱动UI组件基类 - 提供事件驱动UI组件的基础实现

    此类作为事件驱动UI组件的基类，提供与事件系统集成的基础功能。
    派生类可以覆盖方法来实现自定义的UI更新行为。
    支持异步事件处理、批处理优化和自动性能监控。
    """

    def __init__(self, component_name: str):
        """
        初始化事件驱动UI组件

        Args:
            component_name: 组件名称
        """
        self.component_name = component_name
        self.publisher = UIEventPublisher.get_instance(component_name)
        self.subscriber = UIEventSubscriber.get_instance(component_name)

        # 组件状态
        self.active = True
        self.visible = True
        self.enabled = True
        self.needs_update = False

        # 性能数据
        self.last_update_time = 0
        self.update_count = 0
        self.total_update_time = 0

        # 事件记录
        self.event_recorder = None

        self._setup_event_handlers()
        logger.info(f"事件驱动UI组件已初始化: {component_name}")

    def _setup_event_handlers(self):
        """
        设置事件处理器

        派生类应覆盖此方法来设置所需的事件处理器。
        """
        # 基本订阅
        self.subscriber.subscribe_to_notifications(self.handle_notification)
        self.subscriber.subscribe_to_performance_warnings(
            self.handle_performance_warning)

    def enable_event_recording(self, max_events: int = 100) -> None:
        """
        启用事件记录

        Args:
            max_events: 最大记录事件数
        """
        self.event_recorder = EventRecorder(self.component_name, max_events)
        logger.info(f"组件 {self.component_name} 已启用事件记录")

    def disable_event_recording(self) -> None:
        """禁用事件记录"""
        if self.event_recorder:
            self.event_recorder.save()
            self.event_recorder = None
            logger.info(f"组件 {self.component_name} 已禁用事件记录")

    def handle_notification(self, data: Dict[str, Any]) -> None:
        """
        处理通知事件

        Args:
            data: 通知事件数据
        """
        # 基本实现，派生类可以覆盖
        message = data.get('message', '')
        level = data.get('level', 'info')
        logger.info(
            f"UI组件 {self.component_name} 收到通知: [{level}] {message}")

        # 记录事件
        if self.event_recorder:
            self.event_recorder.record_event(UIEventTypes.NOTIFICATION_SHOWN,
                                             data)

    def handle_performance_warning(self, data: Dict[str, Any]) -> None:
        """
        处理性能警告事件

        Args:
            data: 性能警告事件数据
        """
        # 基本实现，派生类可以覆盖
        warning_type = data.get('warning_type', 'unknown')
        logger.warning(
            f"UI组件 {self.component_name} 收到性能警告: {warning_type}")

        # 记录事件
        if self.event_recorder:
            self.event_recorder.record_event(UIEventTypes.PERFORMANCE_WARNING,
                                             data)

    def handle_ui_state_change(self, data: Dict[str, Any]) -> None:
        """
        处理UI状态变更事件

        Args:
            data: UI状态变更事件数据
        """
        # 基本实现，派生类可以覆盖
        state_name = data.get('state_name', '')
        new_value = data.get('new_value', None)
        logger.info(
            f"UI组件 {self.component_name} 状态变更: {state_name} -> {new_value}")

        # 标记需要更新
        self.needs_update = True

        # 记录事件
        if self.event_recorder:
            self.event_recorder.record_event(
                data.get('event_type', UIEventTypes.OPTION_TOGGLED), data)

    def set_visibility(self, visible: bool) -> None:
        """
        设置组件可见性

        Args:
            visible: 是否可见
        """
        if self.visible != visible:
            old_value = self.visible
            self.visible = visible

            # 发布状态变更事件
            self.publisher.publish_ui_state_change(
                UIEventTypes.OPTION_TOGGLED,
                "visibility",
                old_value,
                visible,
                component=self.component_name
            )

    def set_enabled(self, enabled: bool) -> None:
        """
        设置组件启用状态

        Args:
            enabled: 是否启用
        """
        if self.enabled != enabled:
            old_value = self.enabled
            self.enabled = enabled

            # 发布状态变更事件
            self.publisher.publish_ui_state_change(
                UIEventTypes.OPTION_TOGGLED,
                "enabled",
                old_value,
                enabled,
                component=self.component_name
            )

    def update(self, force: bool = False) -> bool:
        """
        更新UI组件

        Args:
            force: 是否强制更新

        Returns:
            bool: 是否成功更新
        """
        # 如果不活跃或不可见且非强制更新，则跳过
        if (not self.active or not self.visible) and not force:
            return False

        # 如果不需要更新且非强制更新，则跳过
        if not self.needs_update and not force:
            return False

        start_time = time.time()

        try:
            # 执行实际更新逻辑，派生类应覆盖_do_update方法
            result = self._do_update(force)

            # 重置更新标志
            self.needs_update = False

            # 更新性能统计
            elapsed = time.time() - start_time
            self.last_update_time = elapsed
            self.update_count += 1
            self.total_update_time += elapsed

            # 发布性能数据
            if elapsed > 0.1:  # 100ms
                self.publisher.publish_performance_warning(
                    "slow_update",
                    {
                        "component": self.component_name,
                        "update_time": elapsed,
                        "average_time": self.get_average_update_time()
                    }
                )

            return result
        except Exception as e:
            logger.error(f"UI组件 {self.component_name} 更新失败: {e}")
            return False

    def _do_update(self, force: bool = False) -> bool:
        """
        执行实际更新逻辑

        Args:
            force: 是否强制更新

        Returns:
            bool: 是否成功更新
        """
        # 基本实现，派生类应覆盖此方法
        return True

    def get_average_update_time(self) -> float:
        """获取平均更新时间"""
        if self.update_count == 0:
            return 0
        return self.total_update_time / self.update_count

    def mark_for_update(self) -> None:
        """标记组件需要更新"""
        self.needs_update = True

    def refresh(self) -> bool:
        """强制刷新组件"""
        return self.update(force=True)

    def request_frame(self) -> None:
        """请求渲染帧"""
        self.publisher.publish_display_event(
            UIEventTypes.RENDER_STARTED,
            component=self.component_name
        )

    def notify(self, message: str, level: str = "info",
               duration: float = 3.0) -> str:
        """
        发送通知

        Args:
            message: 通知消息
            level: 通知级别
            duration: 显示时长

        Returns:
            str: 通知ID
        """
        return self.publisher.publish_notification(
            message,
            level,
            duration,
            component=self.component_name
        )

    def report_performance(self, fps: float = None,
                           frame_time: float = None) -> None:
        """
        报告性能数据

        Args:
            fps: 当前帧率
            frame_time: 帧处理时间(毫秒)
        """
        if fps is not None:
            self.publisher.publish_fps_update(
                fps,
                frame_time,
                component=self.component_name
            )

    def cleanup(self) -> bool:
        """
        清理资源，取消事件订阅

        Returns:
            bool: 是否成功清理
        """
        try:
            self.active = False
            self.visible = False
            self.subscriber.unsubscribe_all()

            # 禁用事件记录
            if self.event_recorder:
                self.disable_event_recording()

            # 发布组件关闭事件
            self.publisher.publish_ui_state_change(
                UIEventTypes.APP_STATE_CHANGED,
                "component_state",
                "active",
                "closed",
                component=self.component_name
            )

            logger.info(f"UI组件 {self.component_name} 已清理资源")
            return True
        except Exception as e:
            logger.error(f"UI组件 {self.component_name} 清理资源失败: {e}")
            return False

# 第5部分结束：事件驱动的UI组件基类
# -*- coding: utf-8 -*-
"""
第6部分：事件记录和回放
"""


class EventRecorder:
    """事件记录器 - 记录UI事件并支持回放"""

    def __init__(self, component_name: str, max_events: int = 100,
                 auto_save: bool = True, save_dir: str = "event_logs"):
        """
        初始化事件记录器

        Args:
            component_name: 组件名称
            max_events: 最大记录事件数
            auto_save: 是否自动保存
            save_dir: 保存目录
        """
        self.component_name = component_name
        self.max_events = max_events
        self.auto_save = auto_save
        self.save_dir = save_dir
        self.events = deque(maxlen=max_events)
        self.start_time = time.time()
        self.recording = True

        # 确保保存目录存在
        if auto_save and not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                logger.error(f"创建事件日志目录失败: {e}")
                self.auto_save = False

        logger.info(f"事件记录器已初始化: {component_name}")

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        记录事件

        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if not self.recording:
            return

        # 创建事件记录
        event_record = {
            "event_type": event_type,
            "timestamp": time.time(),
            "relative_time": time.time() - self.start_time,
            "data": data.copy() if isinstance(data, dict) else {"value": data}
        }

        self.events.append(event_record)

    def pause(self) -> None:
        """暂停记录"""
        self.recording = False

    def resume(self) -> None:
        """恢复记录"""
        self.recording = True

    def clear(self) -> None:
        """清空记录"""
        self.events.clear()
        self.start_time = time.time()

    def get_events(self) -> List[Dict[str, Any]]:
        """获取所有记录的事件"""
        return list(self.events)

    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """获取特定类型的事件"""
        return [e for e in self.events if e["event_type"] == event_type]

    def save(self, filename: str = None) -> str:
        """
        保存事件记录到文件

        Args:
            filename: 文件名，None则自动生成

        Returns:
            str: 保存的文件路径
        """
        if not self.events:
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.component_name}_{timestamp}.json"

        filepath = os.path.join(self.save_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "component": self.component_name,
                    "start_time": self.start_time,
                    "end_time": time.time(),
                    "events_count": len(self.events),
                    "events": list(self.events)
                }, f, indent=2)

            logger.info(f"事件记录已保存: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存事件记录失败: {e}")
            return None

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        加载事件记录

        Args:
            filepath: 文件路径

        Returns:
            Dict: 事件记录数据
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载事件记录失败: {e}")
            return None


class EventPlayer:
    """事件回放器 - 回放记录的事件序列"""

    def __init__(self, publisher: UIEventPublisher = None):
        """
        初始化事件回放器

        Args:
            publisher: 事件发布器，None则创建新的
        """
        self.publisher = publisher or UIEventPublisher("event_player")
        self.events = []
        self.playing = False
        self.speed = 1.0  # 回放速度倍率
        self.start_time = 0
        self.current_index = 0
        self.play_thread = None

    def load_events(self, events_data: Union[str, Dict, List]) -> bool:
        """
        加载事件数据

        Args:
            events_data: 事件数据文件路径或数据对象

        Returns:
            bool: 是否成功加载
        """
        try:
            if isinstance(events_data, str):
                # 加载文件
                data = EventRecorder.load(events_data)
                if not data:
                    return False
                self.events = data.get("events", [])
            elif isinstance(events_data, dict):
                # 直接使用字典数据
                self.events = events_data.get("events", [])
            elif isinstance(events_data, list):
                # 直接使用列表数据
                self.events = events_data
            else:
                logger.error("不支持的事件数据格式")
                return False

            # 按时间排序
            self.events.sort(key=lambda e: e.get("relative_time", 0))
            self.current_index = 0

            logger.info(f"已加载 {len(self.events)} 个事件")
            return len(self.events) > 0
        except Exception as e:
            logger.error(f"加载事件数据失败: {e}")
            return False

    def play(self, speed: float = 1.0) -> None:
        """
        开始回放事件

        Args:
            speed: 回放速度倍率
        """
        if self.playing or not self.events:
            return

        self.playing = True
        self.speed = speed
        self.start_time = time.time()
        self.current_index = 0

        # 在单独线程中回放
        self.play_thread = threading.Thread(
            target=self._play_loop,
            name="event_player",
            daemon=True
        )
        self.play_thread.start()

        logger.info(f"开始回放事件，速度: {speed}x")

    def _play_loop(self) -> None:
        """事件回放循环"""
        try:
            while self.playing and self.current_index < len(self.events):
                # 获取当前事件
                event = self.events[self.current_index]
                relative_time = event.get("relative_time", 0)

                # 计算应该等待的时间
                elapsed = (time.time() - self.start_time) * self.speed
                wait_time = relative_time - elapsed

                if wait_time > 0:
                    # 还未到该事件的时间，等待
                    time.sleep(wait_time / self.speed)

                # 发布事件
                if self.playing:  # 再次检查，可能在等待过程中停止了
                    self._publish_event(event)
                    self.current_index += 1

            # 回放完成
            if self.playing and self.current_index >= len(self.events):
                logger.info("事件回放完成")
                self.playing = False
        except Exception as e:
            logger.error(f"事件回放出错: {e}")
            self.playing = False

    def _publish_event(self, event: Dict[str, Any]) -> None:
        """
        发布单个事件

        Args:
            event: 事件数据
        """
        event_type = event.get("event_type")
        data = event.get("data", {})

        # 更新时间戳为当前时间
        if isinstance(data, dict):
            data["timestamp"] = time.time()
            data["replayed"] = True

        # 发布事件
        self.publisher.publish(event_type, data)
        logger.debug(f"回放事件: {event_type}")

    def pause(self) -> None:
        """暂停回放"""
        self.playing = False

    def resume(self) -> None:
        """恢复回放"""
        if not self.playing and self.current_index < len(self.events):
            # 重新计算开始时间，使回放从当前事件继续
            current_event = self.events[self.current_index]
            relative_time = current_event.get("relative_time", 0)
            self.start_time = time.time() - (relative_time / self.speed)
            self.play(self.speed)

    def stop(self) -> None:
        """停止回放"""
        self.playing = False
        self.current_index = 0

    def set_speed(self, speed: float) -> None:
        """
        设置回放速度

        Args:
            speed: 速度倍率
        """
        if speed <= 0:
            logger.warning("回放速度必须大于0")
            return

        if self.playing:
            # 调整开始时间，使当前事件进度不变
            current_elapsed = (time.time() - self.start_time) * self.speed
            self.start_time = time.time() - (current_elapsed / speed)

        self.speed = speed
        logger.info(f"回放速度已设置为: {speed}x")

    def jump_to(self, time_position: float) -> None:
        """
        跳转到指定时间位置

        Args:
            time_position: 目标时间位置(秒)
        """
        # 找到目标时间点之前的最后一个事件
        target_index = 0
        for i, event in enumerate(self.events):
            if event.get("relative_time", 0) <= time_position:
                target_index = i
            else:
                break

        self.current_index = target_index

        if self.playing:
            # 调整开始时间，使回放从目标位置继续
            self.start_time = time.time() - (time_position / self.speed)

        logger.info(
            f"已跳转到: {time_position:.2f}秒, 事件索引: {target_index}")

# 第6部分结束：事件记录和回放
# -*- coding: utf-8 -*-
"""
第7部分：实用UI组件 (通知管理器)
"""


class NotificationManager(UIEventDrivenComponent):
    """通知管理器 - 处理系统通知的显示和管理"""

    def __init__(self, max_visible: int = 5, default_duration: float = 3.0):
        """
        初始化通知管理器

        Args:
            max_visible: 最大可见通知数
            default_duration: 默认显示时长
        """
        super().__init__("notification_manager")
        self.notifications = {}  # id -> notification
        self.notification_queue = deque()
        self.max_visible = max_visible
        self.default_duration = default_duration
        self.visible_count = 0

        # 订阅事件
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """设置事件处理器"""
        super()._setup_event_handlers()

        self.subscriber.subscribe(
            UIEventTypes.NOTIFICATION_SHOWN,
            self.handle_notification_shown
        )

        self.subscriber.subscribe(
            UIEventTypes.NOTIFICATION_CLICKED,
            self.handle_notification_clicked
        )

        self.subscriber.subscribe(
            UIEventTypes.NOTIFICATION_CLOSED,
            self.handle_notification_closed
        )

    def handle_notification_shown(self, data: Dict[str, Any]) -> None:
        """
        处理通知显示事件

        Args:
            data: 通知数据
        """
        notification_id = data.get("notification_id", str(uuid.uuid4()))
        message = data.get("message", "")
        level = data.get("level", "info")
        duration = data.get("duration", self.default_duration)

        # 创建通知对象
        notification = {
            "id": notification_id,
            "message": message,
            "level": level,
            "duration": duration,
            "created_at": time.time(),
            "visible": False,
            "component": data.get("component", "unknown")
        }

        # 保存通知
        self.notifications[notification_id] = notification

        # 尝试显示通知
        self._show_next_notifications()

        logger.debug(f"通知已添加: {message}")

    def handle_notification_clicked(self, data: Dict[str, Any]) -> None:
        """
        处理通知点击事件

        Args:
            data: 通知数据
        """
        notification_id = data.get("notification_id")
        if not notification_id or notification_id not in self.notifications:
            return

        # 获取通知
        notification = self.notifications[notification_id]

        # 如果通知有关联动作，执行它
        action = notification.get("action")
        if action:
            try:
                if callable(action):
                    action(notification)
                elif isinstance(action, dict) and "event_type" in action:
                    # 发布关联事件
                    self.publisher.publish(
                        action["event_type"],
                        action.get("data", {})
                    )
            except Exception as e:
                logger.error(f"执行通知动作失败: {e}")

        logger.debug(f"通知被点击: {notification_id}")

    def handle_notification_closed(self, data: Dict[str, Any]) -> None:
        """
        处理通知关闭事件

        Args:
            data: 通知数据
        """
        notification_id = data.get("notification_id")
        if not notification_id or notification_id not in self.notifications:
            return

        # 移除通知
        notification = self.notifications.pop(notification_id)

        if notification.get("visible", False):
            self.visible_count -= 1

        # 显示下一个通知
        self._show_next_notifications()

        logger.debug(f"通知已关闭: {notification_id}")

    def _show_next_notifications(self) -> None:
        """显示等待队列中的下一个通知"""
        # 检查是否有空间显示更多通知
        while self.visible_count < self.max_visible:
            # 首先检查已有通知中是否有未显示的
            pending = [n for n in self.notifications.values() if
                       not n.get("visible", False)]

            if not pending:
                break

            # 按创建时间排序，优先显示较早的通知
            notification = min(pending, key=lambda n: n.get("created_at", 0))

            # 标记为可见
            notification["visible"] = True
            self.visible_count += 1

            # 发布通知UI更新事件
            self.publisher.publish_ui_state_change(
                UIEventTypes.NOTIFICATION_SHOWN,
                "notification_visibility",
                False,
                True,
                notification_id=notification["id"],
                notification=notification
            )

            # 设置自动关闭计时器
            duration = notification.get("duration", self.default_duration)
            if duration > 0:
                def close_notification():
                    self.close_notification(notification["id"])

                threading.Timer(duration, close_notification).start()

    def show_notification(self, message: str, level: str = "info",
                          duration: float = None, action=None) -> str:
        """
        显示新通知

        Args:
            message: 通知消息
            level: 级别 (info, warning, error)
            duration: 显示时长，None使用默认值
            action: 点击动作

        Returns:
            str: 通知ID
        """
        duration = duration if duration is not None else self.default_duration

        # 构建通知数据
        data = {
            "message": message,
            "level": level,
            "duration": duration,
            "component": self.component_name
        }

        if action:
            data["action"] = action

        # 发布通知事件
        notification_id = self.publisher.publish_notification(**data)
        return notification_id

    def close_notification(self, notification_id: str) -> bool:
        """
        关闭指定通知

        Args:
            notification_id: 通知ID

        Returns:
            bool: 是否成功关闭
        """
        if notification_id not in self.notifications:
            return False

        # 发布通知关闭事件
        self.publisher.publish(
            UIEventTypes.NOTIFICATION_CLOSED,
            {
                "notification_id": notification_id,
                "component": self.component_name
            }
        )

        return True

    def close_all_notifications(self) -> int:
        """
        关闭所有通知

        Returns:
            int: 关闭的通知数量
        """
        notification_ids = list(self.notifications.keys())
        count = 0

        for notification_id in notification_ids:
            if self.close_notification(notification_id):
                count += 1

        return count

    def _do_update(self, force: bool = False) -> bool:
        """更新通知显示"""
        # 检查是否有过期的通知
        now = time.time()
        expired_ids = []

        for notification_id, notification in self.notifications.items():
            if notification.get("visible", False):
                created_at = notification.get("created_at", 0)
                duration = notification.get("duration", self.default_duration)

                if duration > 0 and now - created_at > duration:
                    expired_ids.append(notification_id)

        # 关闭过期通知
        for notification_id in expired_ids:
            self.close_notification(notification_id)

        return True

    def cleanup(self) -> bool:
        """清理资源"""
        self.close_all_notifications()
        return super().cleanup()

# 第7部分结束：实用UI组件 (通知管理器)
# -*- coding: utf-8 -*-
"""
第8部分：实用UI组件 (性能监视器)
"""


class PerformanceMonitor(UIEventDrivenComponent):
    """性能监视器 - 监控和报告UI性能指标"""

    def __init__(self, update_interval: float = 1.0,
                 warning_threshold: float = 30.0):
        """
        初始化性能监视器

        Args:
            update_interval: 更新间隔(秒)
            warning_threshold: FPS警告阈值
        """
        super().__init__("performance_monitor")
        self.update_interval = update_interval
        self.warning_threshold = warning_threshold
        self.frame_times = deque(maxlen=60)  # 存储最近60帧的时间
        self.frame_count = 0
        self.last_update_time = time.time()
        self.last_frame_time = self.last_update_time
        self.current_fps = 0
        self.avg_frame_time = 0
        self.event_stats = {}
        self.monitoring_active = True
        self.update_timer = None

        # 开始定时更新
        self._schedule_update()

    def _setup_event_handlers(self):
        """设置事件处理器"""
        super()._setup_event_handlers()

        # 订阅帧渲染事件
        self.subscriber.subscribe(
            UIEventTypes.FRAME_RENDERED,
            self.handle_frame_rendered
        )

        # 订阅所有性能警告
        self.subscriber.subscribe_to_performance_warnings(
            self.handle_performance_warning,
            priority=Priority.HIGH
        )

    def handle_frame_rendered(self, data: Dict[str, Any]) -> None:
        """
        处理帧渲染事件

        Args:
            data: 帧数据
        """
        now = time.time()
        frame_time = (now - self.last_frame_time) * 1000  # 毫秒

        # 记录帧时间
        self.frame_times.append(frame_time)
        self.frame_count += 1
        self.last_frame_time = now

        # 收集发布器组件的性能数据
        if "component" in data and isinstance(data.get("frame_time"),
                                              (int, float)):
            component = data.get("component")
            if component not in self.event_stats:
                self.event_stats[component] = {
                    "frame_times": deque(maxlen=30),
                    "event_counts": {}
                }

            self.event_stats[component]["frame_times"].append(
                data.get("frame_time"))

    def _schedule_update(self) -> None:
        """调度定时更新"""
        if not self.monitoring_active:
            return

        if self.update_timer:
            self.update_timer.cancel()

        self.update_timer = threading.Timer(self.update_interval,
                                            self._timed_update)
        self.update_timer.daemon = True
        self.update_timer.start()

    def _timed_update(self) -> None:
        """定时更新性能指标"""
        self.update()
        self._schedule_update()

    def _do_update(self, force: bool = False) -> bool:
        """更新性能指标"""
        now = time.time()
        elapsed = now - self.last_update_time

        if elapsed < self.update_interval and not force:
            return False

        # 计算FPS
        if self.frame_count > 0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
        else:
            self.current_fps = 0

        # 计算平均帧时间
        if self.frame_times:
            self.avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        else:
            self.avg_frame_time = 0

        # 发布性能更新事件
        self.publisher.publish_fps_update(
            self.current_fps,
            self.avg_frame_time,
            component_stats=self.event_stats
        )

        # 检查是否需要发出警告
        if self.current_fps > 0 and self.current_fps < self.warning_threshold:
            self.publisher.publish_performance_warning(
                "low_fps",
                {
                    "fps": self.current_fps,
                    "frame_time": self.avg_frame_time,
                    "threshold": self.warning_threshold
                }
            )

        self.last_update_time = now
        return True

    def start_monitoring(self) -> None:
        """开始性能监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.last_update_time = time.time()
            self.last_frame_time = self.last_update_time
            self.frame_count = 0
            self._schedule_update()
            logger.info("性能监控已启动")

    def stop_monitoring(self) -> None:
        """停止性能监控"""
        self.monitoring_active = False
        if self.update_timer:
            self.update_timer.cancel()
            self.update_timer = None
        logger.info("性能监控已停止")

    def get_performance_report(self) -> Dict[str, Any]:
        """
        获取性能报告

        Returns:
            Dict: 性能数据
        """
        return {
            "fps": self.current_fps,
            "frame_time": self.avg_frame_time,
            "frame_time_history": list(self.frame_times),
            "component_stats": {
                component: {
                    "avg_frame_time": sum(stats["frame_times"]) / len(
                        stats["frame_times"]) if stats["frame_times"] else 0,
                    "event_counts": stats["event_counts"]
                }
                for component, stats in self.event_stats.items()
            }
        }

    def register_component(self, component_name: str) -> None:
        """
        注册要监控的组件

        Args:
            component_name: 组件名称
        """
        if component_name not in self.event_stats:
            self.event_stats[component_name] = {
                "frame_times": deque(maxlen=30),
                "event_counts": {}
            }

    def cleanup(self) -> bool:
        """清理资源"""
        self.stop_monitoring()
        return super().cleanup()

# 第8部分结束：实用UI组件 (性能监视器)
# -*- coding: utf-8 -*-
"""
第9部分：辅助函数和工具
"""


def create_ui_condition(component: str = None, **kwargs) -> Condition:
    """
    创建UI事件条件

    Args:
        component: 组件名称
        **kwargs: 其他条件参数

    Returns:
        Condition: 条件对象
    """
    condition = Condition()

    if component:
        condition.equals('component', component)

    for key, value in kwargs.items():
        condition.equals(key, value)

    return condition


def get_publisher(component_name: str = None) -> UIEventPublisher:
    """
    获取UI事件发布器实例

    Args:
        component_name: 组件名称

    Returns:
        UIEventPublisher: 发布器实例
    """
    return UIEventPublisher.get_instance(component_name)


def get_subscriber(component_name: str = None) -> UIEventSubscriber:
    """
    获取UI事件订阅器实例

    Args:
        component_name: 组件名称

    Returns:
        UIEventSubscriber: 订阅器实例
    """
    return UIEventSubscriber.get_instance(component_name)


def quick_notify(message: str, level: str = "info", duration: float = 3.0,
                 component: str = "system") -> str:
    """
    快速发送通知

    Args:
        message: 通知消息
        level: 通知级别
        duration: 显示时长
        component: 组件名称

    Returns:
        str: 通知ID
    """
    publisher = get_publisher(component)
    return publisher.publish_notification(message, level, duration)


def setup_batch_policies(component_name: str = None,
                         policies: Dict[str, BatchingPolicy] = None,
                         interval: float = 0.05) -> None:
    """
    为组件设置批处理策略

    Args:
        component_name: 组件名称
        policies: 事件类型到批处理策略的映射
        interval: 批处理间隔
    """
    publisher = get_publisher(component_name)
    publisher.enable_batching(interval)

    if policies:
        for event_type, policy in policies.items():
            publisher.set_batching_policy(event_type, policy)


def create_performance_monitor(
        update_interval: float = 1.0) -> PerformanceMonitor:
    """
    创建并启动性能监视器

    Args:
        update_interval: 更新间隔

    Returns:
        PerformanceMonitor: 监视器实例
    """
    monitor = PerformanceMonitor(update_interval)
    return monitor


def create_notification_manager(max_visible: int = 5) -> NotificationManager:
    """
    创建通知管理器

    Args:
        max_visible: 最大可见通知数

    Returns:
        NotificationManager: 管理器实例
    """
    return NotificationManager(max_visible)


def record_ui_events(component_name: str,
                     max_events: int = 100) -> EventRecorder:
    """
    开始记录UI事件

    Args:
        component_name: 组件名称
        max_events: 最大记录事件数

    Returns:
        EventRecorder: 记录器实例
    """
    recorder = EventRecorder(component_name, max_events)
    return recorder


def replay_ui_events(events_data: Union[str, Dict, List],
                     speed: float = 1.0) -> EventPlayer:
    """
    回放UI事件

    Args:
        events_data: 事件数据文件路径或数据对象
        speed: 回放速度

    Returns:
        EventPlayer: 回放器实例
    """
    player = EventPlayer()
    if player.load_events(events_data):
        player.play(speed)
    return player


def collect_system_performance() -> Dict[str, Any]:
    """
    收集系统性能数据

    Returns:
        Dict: 系统性能数据
    """
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        data = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used": memory.used,
            "memory_total": memory.total,
        }

        # 发布系统性能数据
        publisher = get_publisher("system_monitor")
        publisher.publish(UIEventTypes.SYSTEM_INFO_UPDATED, {
            "system_stats": data,
            "timestamp": time.time()
        })

        return data
    except ImportError:
        logger.warning("无法导入psutil库，系统性能数据收集不可用")
        return {"error": "psutil不可用"}


# 使用示例
def example_basic_usage():
    """基本用法示例"""
    # 创建发布器和订阅器
    publisher = UIEventPublisher("example_component")
    subscriber = UIEventSubscriber("example_component")

    # 定义事件处理函数
    def handle_button_click(data):
        print(f"按钮被点击: {data.get('control_id')}")

    # 订阅事件
    subscriber.subscribe(UIEventTypes.BUTTON_CLICKED, handle_button_click)

    # 发布事件
    publisher.publish_control_event(
        "button",
        "submit_button",
        True,
        UIEventTypes.BUTTON_CLICKED
    )

    # 清理资源
    subscriber.unsubscribe_all()


def example_notification_system():
    """通知系统示例"""
    # 创建通知管理器
    notification_manager = NotificationManager()

    # 显示不同类型的通知
    notification_manager.show_notification("操作已完成", "info")
    notification_manager.show_notification("请注意可能的问题", "warning")
    notification_manager.show_notification("发生错误，请重试", "error")

    # 带动作的通知
    def notification_action(notification):
        print(f"通知动作执行: {notification.get('message')}")

    notification_manager.show_notification(
        "点击查看详情",
        "info",
        duration=5.0,
        action=notification_action
    )

    # 清理资源
    notification_manager.cleanup()


def example_async_events():
    """异步事件处理示例"""
    subscriber = UIEventSubscriber("async_example")
    publisher = UIEventPublisher("async_example")

    # 启用异步处理
    subscriber.enable_async_processing()

    # 定义耗时处理函数
    def heavy_processing(data):
        print("开始耗时处理...")
        time.sleep(2)  # 模拟耗时操作
        print(f"处理完成: {data.get('value')}")

    # 异步订阅
    subscriber.subscribe_async(
        UIEventTypes.DATA_PROCESSING_STARTED,
        heavy_processing
    )

    # 发布多个事件
    for i in range(5):
        publisher.publish(
            UIEventTypes.DATA_PROCESSING_STARTED,
            {"value": i, "timestamp": time.time()}
        )
        print(f"已发布事件 {i}")

    # 等待异步处理完成
    time.sleep(3)

    # 清理资源
    subscriber.cleanup()


def example_performance_monitoring():
    """性能监控示例"""
    # 创建性能监视器
    monitor = PerformanceMonitor(update_interval=0.5)

    # 模拟帧渲染
    def simulate_frames():
        for i in range(50):
            time.sleep(0.03)  # 约33fps
            monitor.publisher.publish_display_event(
                UIEventTypes.FRAME_RENDERED,
                frame_info={"frame_number": i},
                frame_time=30  # 30ms
            )

    # 启动模拟线程
    simulation_thread = threading.Thread(target=simulate_frames)
    simulation_thread.daemon = True
    simulation_thread.start()

    # 等待一段时间
    time.sleep(2)

    # 获取性能报告
    report = monitor.get_performance_report()
    print(f"FPS: {report['fps']:.2f}")
    print(f"平均帧时间: {report['frame_time']:.2f}ms")

    # 等待模拟完成
    simulation_thread.join()

    # 清理资源
    monitor.cleanup()


# 如果作为主模块运行，执行示例
if __name__ == "__main__":
    print("UI事件系统示例")
    print("==============")

    print("\n基本用法:")
    example_basic_usage()

    print("\n通知系统:")
    example_notification_system()

    print("\n异步事件处理:")
    example_async_events()

    print("\n性能监控:")
    example_performance_monitoring()

# 第9部分结束：辅助函数和工具
