# -*- coding: utf-8 -*-
"""
事件系统模块 - 提供基于发布/订阅模式的事件系统

本模块提供:
1. 基于观察者模式的事件系统
2. 支持事件过滤和优先级
3. 线程安全的事件分发
4. 事件历史记录和回放功能
5. 针对UI事件的优化处理
"""
import time
import threading
import logging
import queue
from collections import defaultdict, deque
import functools
import hashlib
import json

# 定义事件类别常量
EVENT_CATEGORY_SYSTEM = "system"  # 系统事件
EVENT_CATEGORY_UI = "ui"  # UI事件
EVENT_CATEGORY_DATA = "data"  # 数据事件
EVENT_CATEGORY_USER = "user"  # 用户事件

# 定义常见UI事件类型
UI_EVENT_CLICK = "click"
UI_EVENT_HOVER = "hover"
UI_EVENT_DRAG = "drag"
UI_EVENT_KEY_PRESS = "key_pressed"
UI_EVENT_DISPLAY_UPDATE = "display_updated"
UI_EVENT_FEATURE_TOGGLE = "feature_toggled"

# 配置日志记录器
try:
    from utils.logger_config import setup_logger

    logger = setup_logger("EventSystem")
except ImportError:
    # 备用日志配置
    logger = logging.getLogger("EventSystem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# 尝试导入事件序列化模块
try:
    from utils.event_serializer import serialize_event, deserialize_event
except ImportError:
    # 基本序列化实现
    def serialize_event(event_type, event_data):
        """基本事件序列化"""
        return {
            'event_type': event_type,
            'timestamp': event_data.get('timestamp', time.time()),
            'data': event_data
        }


    def deserialize_event(serialized_event):
        """基本事件反序列化"""
        return serialized_event['event_type'], serialized_event['data']


class EventSystem:
    """
    基本事件系统 - 提供发布/订阅功能

    支持事件订阅、发布和基本的事件历史记录
    """

    def __init__(self, history_capacity=100, enable_logging=True):
        """
        初始化事件系统

        Args:
            history_capacity: 一般事件的历史记录容量
            enable_logging: 是否启用事件日志
        """
        # 订阅字典：事件类型 -> 订阅者列表
        self.subscribers = defaultdict(list)

        # 线程安全锁
        self.lock = threading.RLock()

        # 事件历史记录 - 按类别分开存储
        self.history_capacity = {
            EVENT_CATEGORY_SYSTEM: history_capacity,
            EVENT_CATEGORY_UI: 20,  # UI事件历史较小
            EVENT_CATEGORY_DATA: history_capacity,
            EVENT_CATEGORY_USER: history_capacity
        }

        self.event_history = {
            EVENT_CATEGORY_SYSTEM: deque(
                maxlen=self.history_capacity[EVENT_CATEGORY_SYSTEM]),
            EVENT_CATEGORY_UI: deque(
                maxlen=self.history_capacity[EVENT_CATEGORY_UI]),
            EVENT_CATEGORY_DATA: deque(
                maxlen=self.history_capacity[EVENT_CATEGORY_DATA]),
            EVENT_CATEGORY_USER: deque(
                maxlen=self.history_capacity[EVENT_CATEGORY_USER])
        }

        # 事件统计信息
        self.event_stats = defaultdict(int)
        self.event_timings = defaultdict(list)  # 用于性能分析

        # 开启日志
        self.enable_logging = enable_logging

        # 事件类型映射表 - 映射事件类型到类别
        self.event_category_map = self._initialize_event_category_map()

        # UI事件批处理
        self.ui_batch_enabled = True
        self.ui_batch_size = 10  # 默认批处理大小
        self.ui_batch_interval = 0.05  # 默认批处理间隔（秒）
        self.ui_batch_queue = deque()
        self.ui_batch_lock = threading.RLock()
        self.ui_batch_timer = None

        # 性能监控
        self.performance_monitoring = False
        self.performance_stats = {
            'total_events': 0,
            'ui_events': 0,
            'processing_time': 0,
            'avg_processing_time': 0
        }

        logger.info(
            f"事件系统初始化: 历史容量={history_capacity}, 日志={enable_logging}")

    def _initialize_event_category_map(self):
        """初始化事件类型到类别的映射"""
        mapping = {}

        # UI事件
        ui_events = [
            UI_EVENT_CLICK, UI_EVENT_HOVER, UI_EVENT_DRAG,
            UI_EVENT_KEY_PRESS, UI_EVENT_DISPLAY_UPDATE,
            UI_EVENT_FEATURE_TOGGLE
        ]
        for event in ui_events:
            mapping[event] = EVENT_CATEGORY_UI

        # 系统事件
        system_events = [
            "system_startup", "system_shutdown", "config_changed",
            "resource_warning", "resource_critical", "cache_hit", "cache_miss"
        ]
        for event in system_events:
            mapping[event] = EVENT_CATEGORY_SYSTEM

        # 数据事件
        data_events = [
            "data_loaded", "data_saved", "data_processed",
            "frame_captured", "detection_failed"
        ]
        for event in data_events:
            mapping[event] = EVENT_CATEGORY_DATA

        # 用户事件
        user_events = [
            "user_login", "user_logout", "user_action",
            "person_detected", "action_recognized", "position_mapped"
        ]
        for event in user_events:
            mapping[event] = EVENT_CATEGORY_USER

        return mapping

    def get_event_category(self, event_type):
        """
        获取事件类型的类别

        Args:
            event_type: 事件类型

        Returns:
            str: 事件类别
        """
        # 先检查精确匹配
        if event_type in self.event_category_map:
            return self.event_category_map[event_type]

        # 然后检查前缀匹配
        for prefix, category in [
            (UI_EVENT_CLICK, EVENT_CATEGORY_UI),
            (UI_EVENT_HOVER, EVENT_CATEGORY_UI),
            (UI_EVENT_DRAG, EVENT_CATEGORY_UI),
            (UI_EVENT_KEY_PRESS, EVENT_CATEGORY_UI),
            ("system_", EVENT_CATEGORY_SYSTEM),
            ("data_", EVENT_CATEGORY_DATA),
            ("user_", EVENT_CATEGORY_USER)
        ]:
            if event_type.startswith(prefix):
                # 记住这个映射以加速将来的查找
                self.event_category_map[event_type] = category
                return category

        # 默认为系统事件
        return EVENT_CATEGORY_SYSTEM

    def set_ui_batch_processing(self, enabled, batch_size=None,
                                batch_interval=None):
        """
        配置UI事件批处理

        Args:
            enabled: 是否启用批处理
            batch_size: 批处理大小
            batch_interval: 批处理间隔（秒）
        """
        self.ui_batch_enabled = enabled

        if batch_size is not None:
            self.ui_batch_size = max(1, batch_size)

        if batch_interval is not None:
            self.ui_batch_interval = max(0.01, batch_interval)

        if self.enable_logging:
            status = "启用" if enabled else "禁用"
            logger.info(
                f"UI事件批处理已{status}, 批大小={self.ui_batch_size}, 间隔={self.ui_batch_interval}秒")

    def set_history_capacity(self, category, capacity):
        """
        设置特定类别事件的历史容量

        Args:
            category: 事件类别
            capacity: 新容量
        """
        if category in self.history_capacity:
            old_capacity = self.history_capacity[category]
            self.history_capacity[category] = max(1, capacity)

            # 更新历史记录队列
            self.event_history[category] = deque(
                list(self.event_history[category])[-capacity:]
                if capacity < old_capacity else
                list(self.event_history[category]),
                maxlen=capacity
            )

            if self.enable_logging:
                logger.info(
                    f"{category}事件历史容量已更新: {old_capacity} -> {capacity}")

    def subscribe(self, event_type, handler):
        """
        订阅事件

        Args:
            event_type: 事件类型
            handler: 处理函数

        Returns:
            bool: 是否成功订阅
        """
        return self.subscribe_with_priority(event_type, handler, 0)

    def subscribe_with_filter(self, event_type, handler, filter_func):
        """
        使用过滤器订阅事件

        Args:
            event_type: 事件类型
            handler: 处理函数
            filter_func: 过滤函数

        Returns:
            bool: 是否成功订阅
        """
        return self.subscribe_with_priority(event_type, handler, 0, filter_func)

    def subscribe_with_priority(self, event_type, handler, priority=0,
                                filter_func=None):
        """
        带优先级的事件订阅

        Args:
            event_type: 事件类型
            handler: 处理函数
            priority: 优先级（数字越大优先级越高）
            filter_func: 过滤函数

        Returns:
            bool: 是否成功订阅
        """
        with self.lock:
            # 检查是否已经订阅
            for subscription in self.subscribers[event_type]:
                if subscription['handler'] == handler:
                    # 更新现有订阅
                    subscription['priority'] = priority
                    subscription['filter'] = filter_func

                    # 重新排序订阅者
                    self._sort_subscribers(event_type)

                    if self.enable_logging:
                        logger.info(
                            f"更新订阅: {event_type}, 处理器: {handler.__name__}, 优先级: {priority}")

                    return True

            # 添加新订阅
            subscription = {
                'handler': handler,
                'filter': filter_func,
                'priority': priority,
                'stats': {
                    'calls': 0,
                    'last_call': 0,
                    'total_time': 0
                }
            }

            self.subscribers[event_type].append(subscription)

            # 按优先级排序
            self._sort_subscribers(event_type)

            if self.enable_logging:
                filter_info = ", 带过滤器" if filter_func else ""
                logger.info(
                    f"新订阅: {event_type}, 处理器: {handler.__name__}, 优先级: {priority}{filter_info}")

            return True

    def _sort_subscribers(self, event_type):
        """对订阅者按优先级排序"""
        self.subscribers[event_type].sort(
            key=lambda sub: sub['priority'],
            reverse=True  # 高优先级排在前面
        )

    def unsubscribe(self, event_type, handler):
        """
        取消订阅

        Args:
            event_type: 事件类型
            handler: 处理函数

        Returns:
            bool: 是否成功取消
        """
        with self.lock:
            if event_type not in self.subscribers:
                return False

            # 查找并移除订阅
            for i, subscription in enumerate(self.subscribers[event_type]):
                if subscription['handler'] == handler:
                    self.subscribers[event_type].pop(i)

                    if self.enable_logging:
                        logger.info(
                            f"取消订阅: {event_type}, 处理器: {handler.__name__}")

                    return True

            return False

    def publish(self, event_type, data=None):
        """
        发布事件

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            bool: 是否成功发布
        """
        start_time = time.time() if self.performance_monitoring else 0

        # 规范化数据
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            data = {'value': data}

        # 添加元数据
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        if 'event_type' not in data:
            data['event_type'] = event_type

        # 确定事件类别
        category = self.get_event_category(event_type)

        # 记录事件历史
        self._record_event(event_type, data, category)

        # 更新事件统计
        with self.lock:
            self.event_stats[event_type] += 1
            self.performance_stats['total_events'] += 1
            if category == EVENT_CATEGORY_UI:
                self.performance_stats['ui_events'] += 1

        # 记录日志
        if self.enable_logging:
            logger.debug(f"发布事件: {event_type}")

        # 处理事件 - UI事件特殊处理
        result = False
        if category == EVENT_CATEGORY_UI and self.ui_batch_enabled:
            # UI事件批处理
            with self.ui_batch_lock:
                # 将事件添加到批处理队列
                self.ui_batch_queue.append((event_type, data))
                queue_size = len(self.ui_batch_queue)

                # 如果达到批处理大小，立即处理
                if queue_size >= self.ui_batch_size:
                    self._process_ui_batch()
                else:
                    # 否则设置定时器
                    self._schedule_ui_batch_processing()

            result = True
        else:
            # 非UI事件直接处理
            result = self._process_event(event_type, data)

        # 计算并记录处理时间
        if self.performance_monitoring:
            processing_time = time.time() - start_time
            self.performance_stats['processing_time'] += processing_time

            # 更新平均处理时间
            total_events = self.performance_stats['total_events']
            if total_events > 0:
                self.performance_stats['avg_processing_time'] = (
                        self.performance_stats['processing_time'] / total_events
                )

            # 记录这个事件类型的处理时间
            self.event_timings[event_type].append(processing_time)
            # 只保留最近的50个时间记录
            if len(self.event_timings[event_type]) > 50:
                self.event_timings[event_type] = self.event_timings[event_type][
                                                 -50:]

        return result

    def _schedule_ui_batch_processing(self):
        """调度UI事件批处理"""
        # 取消现有定时器
        if self.ui_batch_timer:
            self.ui_batch_timer.cancel()

        # 创建新定时器
        self.ui_batch_timer = threading.Timer(
            self.ui_batch_interval,
            self._process_ui_batch
        )
        self.ui_batch_timer.daemon = True
        self.ui_batch_timer.start()

    def _process_ui_batch(self):
        """处理UI事件批次"""
        with self.ui_batch_lock:
            # 如果队列为空，不处理
            if not self.ui_batch_queue:
                return

            # 获取当前批次并清空队列
            batch = list(self.ui_batch_queue)
            self.ui_batch_queue.clear()

        # 处理批次中的每个事件
        for event_type, data in batch:
            self._process_event(event_type, data)

    def _process_event(self, event_type, data):
        """
        处理单个事件

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            bool: 是否有处理器处理了此事件
        """
        with self.lock:
            # 获取订阅者列表副本
            if event_type not in self.subscribers:
                return False

            subscribers = list(self.subscribers[event_type])

        # 是否有任何处理器被调用
        any_processed = False

        # 按优先级顺序调用处理器
        for subscription in subscribers:
            try:
                handler = subscription['handler']
                filter_func = subscription['filter']

                # 应用过滤器
                if filter_func is not None:
                    try:
                        if not filter_func(data):
                            continue  # 不满足过滤条件
                    except Exception as e:
                        logger.error(
                            f"事件过滤器错误: {e}, 事件类型: {event_type}")
                        continue

                # 测量处理时间
                start_time = time.time() if self.performance_monitoring else 0

                # 调用处理器
                handler(data)
                any_processed = True

                # 更新处理器统计
                if self.performance_monitoring:
                    processing_time = time.time() - start_time
                    subscription['stats']['calls'] += 1
                    subscription['stats']['last_call'] = time.time()
                    subscription['stats']['total_time'] += processing_time

            except Exception as e:
                logger.error(
                    f"事件处理错误: {e}, 事件类型: {event_type}, 处理器: {subscription['handler'].__name__}")

        return any_processed

    def _record_event(self, event_type, data, category):
        """记录事件到历史"""
        with self.lock:
            # 添加到相应类别的历史记录
            if category in self.event_history:
                self.event_history[category].append((event_type, data.copy()))

    def get_event_history(self, event_type=None, limit=None, include_ui=False):
        """
        获取事件历史

        Args:
            event_type: 可选的事件类型过滤
            limit: 可选的结果数量限制
            include_ui: 是否包含UI事件

        Returns:
            list: 事件历史列表
        """
        with self.lock:
            all_events = []

            # 从各个类别收集事件
            for category, history in self.event_history.items():
                # 跳过UI事件（如果不包含）
                if category == EVENT_CATEGORY_UI and not include_ui:
                    continue

                all_events.extend(list(history))

            # 按时间戳排序
            all_events.sort(key=lambda x: x[1].get('timestamp', 0))

            # 过滤事件类型
            if event_type:
                all_events = [(t, d) for t, d in all_events if t == event_type]

            # 限制结果数量
            if limit and limit > 0:
                all_events = all_events[-limit:]

            return all_events

    def get_event_stats(self, include_timing=False):
        """
        获取事件统计信息

        Args:
            include_timing: 是否包含事件处理时间统计

        Returns:
            dict: 统计信息
        """
        with self.lock:
            stats = {
                'event_counts': dict(self.event_stats),
                'total_events': sum(self.event_stats.values()),
                'ui_events': sum(
                    self.event_stats.get(et, 0) for et in self.event_stats
                    if self.get_event_category(et) == EVENT_CATEGORY_UI),
                'subscription_counts': {et: len(subs) for et, subs in
                                        self.subscribers.items()},
                'categories': {
                    cat: sum(1 for et in self.event_stats if
                             self.get_event_category(et) == cat)
                    for cat in [EVENT_CATEGORY_SYSTEM, EVENT_CATEGORY_UI,
                                EVENT_CATEGORY_DATA, EVENT_CATEGORY_USER]
                }
            }

            if include_timing and self.performance_monitoring:
                stats['performance'] = {
                    'avg_processing_time': self.performance_stats[
                        'avg_processing_time'],
                    'total_processing_time': self.performance_stats[
                        'processing_time'],
                    'event_avg_times': {
                        et: sum(times) / len(times) if times else 0
                        for et, times in self.event_timings.items()
                    }
                }

            return stats

    def clear_history(self, category=None):
        """
        清除事件历史

        Args:
            category: 可选的类别

        Returns:
            int: 清除的事件数量
        """
        with self.lock:
            if category:
                if category in self.event_history:
                    count = len(self.event_history[category])
                    self.event_history[category].clear()
                    return count
                return 0
            else:
                count = sum(
                    len(history) for history in self.event_history.values())
                for history in self.event_history.values():
                    history.clear()
                return count

    def reset_stats(self):
        """
        重置事件统计

        Returns:
            int: 重置的事件类型数量
        """
        with self.lock:
            count = len(self.event_stats)
            self.event_stats.clear()

            if self.performance_monitoring:
                self.performance_stats = {
                    'total_events': 0,
                    'ui_events': 0,
                    'processing_time': 0,
                    'avg_processing_time': 0
                }
                self.event_timings.clear()

            return count

    def enable_performance_monitoring(self, enabled=True):
        """
        启用或禁用性能监控

        Args:
            enabled: 是否启用
        """
        self.performance_monitoring = enabled

        if enabled:
            # 初始化性能统计
            self.performance_stats = {
                'total_events': 0,
                'ui_events': 0,
                'processing_time': 0,
                'avg_processing_time': 0
            }
            self.event_timings.clear()

        if self.enable_logging:
            status = "启用" if enabled else "禁用"
            logger.info(f"性能监控已{status}")

    def shutdown(self):
        """
        关闭事件系统，清理资源

        此方法会:
        1. 取消所有的UI批处理定时器
        2. 保存最终的事件统计（如果需要）
        3. 释放所有资源
        """
        # 取消UI批处理定时器
        if hasattr(self, 'ui_batch_timer') and self.ui_batch_timer:
            self.ui_batch_timer.cancel()
            self.ui_batch_timer = None

        # 清理UI批处理队列
        if hasattr(self, 'ui_batch_queue'):
            with self.ui_batch_lock:
                self.ui_batch_queue.clear()

        # 记录最后的统计信息到日志
        if self.enable_logging:
            stats = self.get_event_stats()
            self.logger.info(
                f"事件系统关闭 - 处理的事件总数: {sum(stats['event_counts'].values())}")

            if hasattr(self,
                       'performance_monitoring') and self.performance_monitoring:
                perf_stats = self.get_event_stats(include_timing=True).get(
                    'performance', {})
                self.logger.info(
                    f"性能统计 - 平均处理时间: {perf_stats.get('avg_processing_time', 0) * 1000:.3f}毫秒")

        if self.enable_logging:
            self.logger.info("事件系统已关闭")

    def save_event_history(self, filepath, event_type=None, limit=None,
                           include_ui=False):
        """
        将事件历史保存到文件

        Args:
            filepath: 文件路径
            event_type: 可选的事件类型过滤
            limit: 可选的结果数量限制
            include_ui: 是否包含UI事件

        Returns:
            int: 保存的事件数量
        """
        try:
            import os

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 获取事件历史
            events = self.get_event_history(event_type, limit, include_ui)

            # 序列化事件
            serialized_events = []
            for event_type, event_data in events:
                try:
                    serialized_event = serialize_event(event_type, event_data)
                    serialized_events.append(serialized_event)
                except Exception as e:
                    logger.error(f"序列化事件错误: {e}, 事件类型: {event_type}")

            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_events, f, indent=2)

            if self.enable_logging:
                logger.info(
                    f"已保存{len(serialized_events)}个事件到: {filepath}")

            return len(serialized_events)
        except Exception as e:
            logger.error(f"保存事件历史错误: {e}")
            return 0

    def load_event_history(self, filepath):
        """
        从文件加载事件历史

        Args:
            filepath: 文件路径

        Returns:
            list: 加载的事件列表
        """
        try:
            import os

            if not os.path.exists(filepath):
                logger.warning(f"事件历史文件不存在: {filepath}")
                return []

            # 从文件加载
            with open(filepath, 'r', encoding='utf-8') as f:
                serialized_events = json.load(f)

            # 反序列化
            events = []
            for serialized_event in serialized_events:
                try:
                    event_type, event_data = deserialize_event(serialized_event)
                    events.append((event_type, event_data))
                except Exception as e:
                    logger.error(f"反序列化事件错误: {e}")

            if self.enable_logging:
                logger.info(f"已从{filepath}加载{len(events)}个事件")

            return events
        except Exception as e:
            logger.error(f"加载事件历史错误: {e}")
            return []


class EnhancedEventSystem(EventSystem):
    """
    增强版事件系统 - 提供高级事件处理能力

    特性:
    1. 线程安全的事件订阅和发布
    2. 支持事件过滤和优先级
    3. 事件历史记录和统计
    4. 异步事件处理
    5. UI事件批处理和优化
    """

    def __init__(self, history_capacity=100, enable_logging=True,
                 async_mode=False):
        """
        初始化增强版事件系统

        Args:
            history_capacity: 一般事件的历史记录容量
            enable_logging: 是否启用事件日志
            async_mode: 是否启用异步事件处理
        """
        # 初始化基类
        super().__init__(history_capacity, enable_logging)

        # 异步模式设置
        self.async_mode = async_mode

        # 异步处理队列 - 每个类别一个队列
        self.event_queues = {
            EVENT_CATEGORY_SYSTEM: queue.Queue(),
            EVENT_CATEGORY_UI: queue.Queue(),
            EVENT_CATEGORY_DATA: queue.Queue(),
            EVENT_CATEGORY_USER: queue.Queue()
        }

        # 异步处理线程
        self.async_threads = {}
        self.async_running = False

        # 事件优先级队列 - 确保高优先级事件先处理
        self.use_priority_queue = True
        self.priority_queues = {
            # 类别: {优先级: 队列}
            cat: defaultdict(deque) for cat in self.event_queues
        }

        # 启动异步处理
        if async_mode:
            self._start_async_processing()

        if enable_logging:
            logger.info(
                f"增强事件系统初始化: 异步模式={async_mode}, 历史容量={history_capacity}")

    def _start_async_processing(self):
        """启动异步事件处理"""
        if self.async_threads:
            return

        self.async_running = True

        # 为每个类别启动处理线程
        for category in self.event_queues:
            thread = threading.Thread(
                target=self._process_event_queue,
                args=(category,),
                daemon=True
            )
            thread.start()
            self.async_threads[category] = thread

        if self.enable_logging:
            logger.info("异步事件处理线程已启动")

    def _process_event_queue(self, category):
        """
        处理特定类别的事件队列

        Args:
            category: 事件类别
        """
        # 获取相应的队列
        category_queue = self.event_queues[category]

        # 设置超时
        timeout = 0.01 if category == EVENT_CATEGORY_UI else 0.1

        while self.async_running:
            try:
                if self.use_priority_queue:
                    # 优先级队列处理
                    # 检查是否有事件
                    has_events = False
                    with self.lock:
                        for priority_queue in self.priority_queues[
                            category].values():
                            if priority_queue:
                                has_events = True
                                break

                    if not has_events:
                        # 等待新事件
                        time.sleep(timeout)
                        continue

                    # 处理队列中的事件，从高优先级开始
                    with self.lock:
                        # 获取所有优先级，降序排列
                        priorities = sorted(
                            self.priority_queues[category].keys(),
                            reverse=True
                        )

                        # 从最高优先级开始处理
                        for priority in priorities:
                            priority_queue = self.priority_queues[category][
                                priority]
                            if priority_queue:
                                # 获取最早的事件
                                event_type, data = priority_queue.popleft()
                                # 退出锁再处理事件
                                break
                        else:
                            # 没有找到事件，继续等待
                            continue
                else:
                    # 普通队列处理
                    try:
                        # 从队列获取事件
                        event_type, data = category_queue.get(timeout=timeout)
                    except queue.Empty:
                        continue

                # 处理事件
                super()._process_event(event_type, data)

                # 如果使用普通队列，标记任务完成
                if not self.use_priority_queue:
                    category_queue.task_done()

            except Exception as e:
                logger.error(f"处理事件队列错误({category}): {e}")
                time.sleep(0.01)  # 避免CPU占用过高

    def publish(self, event_type, data=None):
        """
        发布事件

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            bool: 是否成功发布
        """
        # 规范化数据
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            data = {'value': data}

        # 添加元数据
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        if 'event_type' not in data:
            data['event_type'] = event_type

        # 确定事件类别
        category = self.get_event_category(event_type)

        # 记录事件历史
        self._record_event(event_type, data, category)

        # 更新事件统计
        with self.lock:
            self.event_stats[event_type] += 1
            if self.performance_monitoring:
                self.performance_stats['total_events'] += 1
                if category == EVENT_CATEGORY_UI:
                    self.performance_stats['ui_events'] += 1

        # 记录日志
        if self.enable_logging:
            logger.debug(f"发布事件: {event_type}")

        # 处理事件
        if self.async_mode:
            # 异步处理
            try:
                if self.use_priority_queue:
                    # 计算事件优先级
                    priority = self._calculate_event_priority(event_type, data)

                    # 添加到优先级队列
                    with self.lock:
                        self.priority_queues[category][priority].append(
                            (event_type, data))
                else:
                    # 添加到普通队列
                    self.event_queues[category].put((event_type, data))

                return True
            except Exception as e:
                logger.error(f"添加事件到队列错误: {e}")
                return False
        else:
            # 同步处理 - UI事件批处理，其他直接处理
            if category == EVENT_CATEGORY_UI and self.ui_batch_enabled:
                with self.ui_batch_lock:
                    # 添加到批处理队列
                    self.ui_batch_queue.append((event_type, data))

                    # 检查是否需要立即处理
                    if len(self.ui_batch_queue) >= self.ui_batch_size:
                        self._process_ui_batch()
                    else:
                        # 设置定时器
                        self._schedule_ui_batch_processing()

                return True
            else:
                # 其他事件直接处理
                return super()._process_event(event_type, data)

    def _calculate_event_priority(self, event_type, data):
        """
        计算事件优先级

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            int: 事件优先级
        """
        # 默认优先级
        priority = 0

        # 检查事件类型
        if event_type.startswith("system_"):
            # 系统事件高优先级
            priority = 100

            # 关键系统事件更高优先级
            if event_type in ["system_shutdown", "system_critical"]:
                priority = 200

        elif event_type.startswith(UI_EVENT_KEY_PRESS):
            # 按键事件较高优先级
            priority = 80

            # 特殊按键更高优先级
            key = data.get('key')
            if key and key in ["Escape", "Enter"]:
                priority = 90

        elif event_type.startswith(UI_EVENT_CLICK):
            # 点击事件普通优先级
            priority = 50

            # 根据元素类型调整优先级
            element_type = data.get('element_type')
            if element_type:
                if element_type == "button":
                    priority = 60
                elif element_type == "form":
                    priority = 70

        elif event_type.startswith(UI_EVENT_HOVER):
            # 悬停事件低优先级
            priority = 20

        # 用户可以在事件数据中指定优先级覆盖
        if 'priority' in data:
            priority = data['priority']

        return priority

    def shutdown(self):
        """关闭增强事件系统"""
        # 停止异步处理
        if self.async_mode and self.async_threads:
            self.async_running = False

            # 等待线程结束
            for category, thread in self.async_threads.items():
                thread.join(timeout=1.0)

            self.async_threads.clear()

            if self.enable_logging:
                logger.info("异步事件处理线程已停止")

        # 调用基类关闭
        super().shutdown()

    def set_use_priority_queue(self, enabled):
        """
        设置是否使用优先级队列

        Args:
            enabled: 是否启用
        """
        self.use_priority_queue = enabled

        if enabled:
            # 清空现有队列内容，防止重复处理
            for category in self.event_queues:
                # 尝试将普通队列中的内容转移到优先级队列
                while not self.event_queues[category].empty():
                    try:
                        event_type, data = self.event_queues[
                            category].get_nowait()
                        priority = self._calculate_event_priority(event_type,
                                                                  data)

                        with self.lock:
                            self.priority_queues[category][priority].append(
                                (event_type, data))

                        self.event_queues[category].task_done()
                    except queue.Empty:
                        break

        else:
            # 清空优先级队列内容
            with self.lock:
                for category in self.priority_queues:
                    for priority_queue in self.priority_queues[
                        category].values():
                        # 将优先级队列中的事件转移到普通队列
                        for event_type, data in priority_queue:
                            self.event_queues[category].put((event_type, data))

                    # 清空优先级队列
                    self.priority_queues[category].clear()

        if self.enable_logging:
            status = "启用" if enabled else "禁用"
            logger.info(f"优先级队列已{status}")

    def get_queue_stats(self):
        """
        获取队列统计信息

        Returns:
            dict: 队列统计
        """
        stats = {}

        if self.async_mode:
            # 普通队列统计
            queue_sizes = {
                category: queue_obj.qsize()
                for category, queue_obj in self.event_queues.items()
            }
            stats['queue_sizes'] = queue_sizes

            # 优先级队列统计
            if self.use_priority_queue:
                priority_stats = {}

                with self.lock:
                    for category, priority_map in self.priority_queues.items():
                        category_stats = {
                            priority: len(queue)
                            for priority, queue in priority_map.items()
                            if queue  # 只统计非空队列
                        }

                        if category_stats:
                            priority_stats[category] = category_stats

                stats['priority_queues'] = priority_stats

        # UI批处理统计
        if self.ui_batch_enabled:
            with self.ui_batch_lock:
                stats['ui_batch_queue_size'] = len(self.ui_batch_queue)

        return stats


# 单例模式
_event_system = None


def get_event_system(history_capacity=100, enable_logging=True,
                     async_mode=False):
    """
    获取全局事件系统实例

    Args:
        history_capacity: 一般事件的历史记录容量
        enable_logging: 是否启用事件日志
        async_mode: 是否启用异步事件处理

    Returns:
        EnhancedEventSystem: 事件系统实例
    """
    global _event_system

    if _event_system is None:
        _event_system = EnhancedEventSystem(
            history_capacity=history_capacity,
            enable_logging=enable_logging,
            async_mode=async_mode
        )

    return _event_system


# 如果直接运行此模块，执行测试代码
if __name__ == "__main__":
    # 测试事件系统
    print("测试事件系统...")

    # 创建事件系统
    events = EnhancedEventSystem(async_mode=False)
    events.enable_performance_monitoring(True)

    # 启用UI事件批处理
    events.set_ui_batch_processing(True, batch_size=5, batch_interval=0.05)


    # 定义处理函数
    def system_handler(data):
        print(f"系统事件处理: {data}")
        time.sleep(0.001)  # 模拟处理时间


    def ui_handler(data):
        print(f"UI事件处理: {data}")


    def critical_handler(data):
        print(f"紧急事件处理: {data}")


    # 订阅事件
    events.subscribe("system_event", system_handler)
    events.subscribe(UI_EVENT_CLICK, ui_handler)
    events.subscribe_with_priority(
        "system_critical",
        critical_handler,
        priority=100  # 高优先级
    )

    # 测试UI事件批处理
    print("\n发布UI事件批次:")
    for i in range(8):
        events.publish(UI_EVENT_CLICK, {
            'element': f"button_{i}",
            'element_type': "button",
            'x': i * 10,
            'y': i * 5
        })
        time.sleep(0.01)

    # 等待批处理完成
    time.sleep(0.1)

    # 测试系统事件
    print("\n发布系统事件:")
    events.publish("system_event", {
        'action': "startup",
        'timestamp': time.time(),
        'status': "success"
    })

    # 测试高优先级事件
    print("\n发布紧急事件:")
    events.publish("system_critical", {
        'message': "系统资源不足",
        'severity': "high"
    })

    # 显示事件统计
    time.sleep(0.1)  # 确保所有事件都已处理
    stats = events.get_event_stats(include_timing=True)

    print("\n事件统计:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count}个事件类型")

    print(f"  总事件数: {stats['total_events']}")
    print(f"  UI事件数: {stats['ui_events']}")

    if 'performance' in stats:
        perf = stats['performance']
        print(f"  平均处理时间: {perf['avg_processing_time'] * 1000:.3f}毫秒")

    # 关闭事件系统
    events.shutdown()
