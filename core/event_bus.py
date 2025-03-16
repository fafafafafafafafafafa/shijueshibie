# -*- coding: utf-8 -*-
"""
中央事件总线模块 - 提供统一的事件发布/订阅机制
支持事件过滤、优先级处理和异步事件处理
"""

import threading
import queue
import time
import logging
import uuid
import weakref
from typing import Dict, List, Callable, Any, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from .event_models import Event, EventType, EventPriority, EventMetadata


# 设置日志
logger = logging.getLogger("EventBus")


class EventBusMode(Enum):
    """事件总线运行模式"""
    SYNCHRONOUS = "sync"  # 同步模式 - 直接在调用线程中执行处理器
    ASYNCHRONOUS = "async"  # 异步模式 - 在后台线程中执行处理器
    HYBRID = "hybrid"  # 混合模式 - 根据事件配置决定同步或异步


class EventFilter:
    """事件过滤器 - 用于筛选感兴趣的事件"""

    def __init__(self,
                 event_types: Optional[Set[EventType]] = None,
                 source_ids: Optional[Set[str]] = None,
                 min_priority: EventPriority = EventPriority.NORMAL,
                 custom_filter: Optional[Callable[[Event], bool]] = None):
        """
        初始化事件过滤器

        Args:
            event_types: 感兴趣的事件类型集合，None表示接受所有类型
            source_ids: 感兴趣的事件源ID集合，None表示接受所有来源
            min_priority: 最小事件优先级，低于此优先级的事件将被过滤
            custom_filter: 自定义过滤函数，返回True表示接受该事件
        """
        self.event_types = event_types
        self.source_ids = source_ids
        self.min_priority = min_priority
        self.custom_filter = custom_filter

    def match(self, event: Event) -> bool:
        """
        检查事件是否匹配过滤条件

        Args:
            event: 要检查的事件对象

        Returns:
            bool: 是否匹配过滤条件
        """
        # 检查事件类型
        if self.event_types is not None and event.event_type not in self.event_types:
            return False

        # 检查事件源
        if self.source_ids is not None and event.metadata.source_id not in self.source_ids:
            return False

        # 检查事件优先级
        if event.metadata.priority.value < self.min_priority.value:
            return False

        # 应用自定义过滤器
        if self.custom_filter is not None and not self.custom_filter(event):
            return False

        return True


@dataclass
class SubscriptionInfo:
    """订阅信息 - 包含处理器和过滤器"""
    subscriber_id: str
    handler: Callable[[Event], None]
    filter: EventFilter
    subscription_time: float = field(default_factory=time.time)

    # 统计信息
    handled_count: int = 0
    last_handled_time: Optional[float] = None
    total_handling_time: float = 0

    def update_stats(self, handling_time: float):
        """更新处理统计信息"""
        self.handled_count += 1
        self.last_handled_time = time.time()
        self.total_handling_time += handling_time


class EventBus:
    """
    中央事件总线 - 处理事件的发布和订阅
    支持同步和异步事件处理，以及事件过滤和优先级
    """

    _instance = None  # 单例实例

    @classmethod
    def get_instance(cls) -> 'EventBus':
        """获取EventBus单例实例"""
        if cls._instance is None:
            cls._instance = EventBus()
        return cls._instance

    def __init__(self, mode: EventBusMode = EventBusMode.HYBRID,
                 worker_count: int = 1):
        """
        初始化事件总线

        Args:
            mode: 事件总线运行模式
            worker_count: 异步工作线程数量
        """
        if EventBus._instance is not None:
            logger.warning("EventBus是单例类，请使用get_instance()获取实例")
            return

        self.mode = mode
        self.worker_count = worker_count

        # 订阅字典 {event_type: [SubscriptionInfo, ...]}
        self.subscriptions: Dict[EventType, List[SubscriptionInfo]] = {}
        # 全局订阅列表 - 接收所有事件
        self.global_subscriptions: List[SubscriptionInfo] = []

        # 事件队列 - 用于异步处理
        self.event_queue = queue.PriorityQueue()

        # 工作线程
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.RLock()

        # 事件和订阅统计
        self.published_events_count = 0
        self.delivered_events_count = 0
        self.dropped_events_count = 0

        # 事件历史（有限容量）
        self.max_history_size = 100
        self.event_history: List[Event] = []

        # 订阅者ID映射 {subscriber_id: {event_type: SubscriptionInfo}}
        self.subscriber_map: Dict[
            str, Dict[Optional[EventType], SubscriptionInfo]] = {}

        logger.info(
            f"事件总线初始化完成，模式: {mode.value}，工作线程: {worker_count}")

    def start(self):
        """启动事件总线（启动工作线程）"""
        with self.lock:
            if self.running:
                return

            self.running = True

            # 创建并启动工作线程
            if self.mode in [EventBusMode.ASYNCHRONOUS, EventBusMode.HYBRID]:
                for i in range(self.worker_count):
                    worker = threading.Thread(
                        target=self._worker_loop,
                        name=f"EventBus-Worker-{i}",
                        daemon=True
                    )
                    worker.start()
                    self.workers.append(worker)

                logger.info(f"事件总线已启动 {len(self.workers)} 个工作线程")

    def stop(self):
        """停止事件总线（停止工作线程）"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            # 等待工作线程完成
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)

            self.workers.clear()
            logger.info("事件总线已停止")

    def subscribe(self,
                  event_type: Optional[EventType],
                  handler: Callable[[Event], None],
                  subscriber_id: Optional[str] = None,
                  filter_: Optional[EventFilter] = None) -> str:
        """
        订阅特定类型的事件

        Args:
            event_type: 要订阅的事件类型，None表示订阅所有事件
            handler: 事件处理函数，接受一个Event参数
            subscriber_id: 订阅者ID，如果为None则自动生成
            filter_: 事件过滤器，用于进一步筛选事件

        Returns:
            str: 订阅者ID，用于后续取消订阅
        """
        with self.lock:
            # 生成订阅者ID（如果未提供）
            if subscriber_id is None:
                subscriber_id = str(uuid.uuid4())

            # 创建订阅信息
            subscription = SubscriptionInfo(
                subscriber_id=subscriber_id,
                handler=handler,
                filter=filter_ or EventFilter()
            )

            # 添加到订阅映射
            if subscriber_id not in self.subscriber_map:
                self.subscriber_map[subscriber_id] = {}
            self.subscriber_map[subscriber_id][event_type] = subscription

            # 添加到事件类型映射
            if event_type is None:
                # 全局订阅 - 接收所有事件
                self.global_subscriptions.append(subscription)
                logger.debug(f"添加全局订阅: {subscriber_id}")
            else:
                # 特定事件类型订阅
                if event_type not in self.subscriptions:
                    self.subscriptions[event_type] = []
                self.subscriptions[event_type].append(subscription)
                logger.debug(
                    f"添加事件订阅: {subscriber_id} -> {event_type.name}")

            return subscriber_id

    def unsubscribe(self, subscriber_id: str,
                    event_type: Optional[EventType] = None) -> bool:
        """
        取消订阅

        Args:
            subscriber_id: 订阅者ID
            event_type: 要取消订阅的事件类型，None表示取消所有订阅

        Returns:
            bool: 是否成功取消订阅
        """
        with self.lock:
            if subscriber_id not in self.subscriber_map:
                return False

            if event_type is None:
                # 取消所有订阅
                for evt_type, subscription in list(
                        self.subscriber_map[subscriber_id].items()):
                    if evt_type is None:
                        # 从全局订阅中移除
                        self.global_subscriptions = [s for s in
                                                     self.global_subscriptions
                                                     if
                                                     s.subscriber_id != subscriber_id]
                    else:
                        # 从特定事件类型订阅中移除
                        if evt_type in self.subscriptions:
                            self.subscriptions[evt_type] = [s for s in
                                                            self.subscriptions[
                                                                evt_type]
                                                            if
                                                            s.subscriber_id != subscriber_id]

                # 清除订阅者映射
                del self.subscriber_map[subscriber_id]
                logger.debug(f"已移除订阅者的所有订阅: {subscriber_id}")
                return True
            else:
                # 仅取消特定事件类型的订阅
                if event_type in self.subscriber_map[subscriber_id]:
                    # 从订阅者映射中移除
                    subscription = self.subscriber_map[subscriber_id].pop(
                        event_type)

                    # 从事件类型映射中移除
                    if event_type in self.subscriptions:
                        self.subscriptions[event_type] = [s for s in
                                                          self.subscriptions[
                                                              event_type]
                                                          if
                                                          s.subscriber_id != subscriber_id]

                    logger.debug(
                        f"已移除订阅: {subscriber_id} -> {event_type.name}")
                    return True
                else:
                    return False

    def publish(self, event: Event, async_mode: Optional[bool] = None) -> str:
        """
        发布事件

        Args:
            event: 要发布的事件对象
            async_mode: 是否异步处理，None表示使用总线默认模式

        Returns:
            str: 事件ID
        """
        # 设置事件ID（如果未设置）
        if not event.metadata.event_id:
            event.metadata.event_id = str(uuid.uuid4())

        # 设置发布时间（如果未设置）
        if not event.metadata.publication_time:
            event.metadata.publication_time = time.time()

        # 更新统计
        self.published_events_count += 1

        # 添加到历史记录
        self._add_to_history(event)

        # 确定处理模式
        use_async = async_mode
        if use_async is None:
            if self.mode == EventBusMode.SYNCHRONOUS:
                use_async = False
            elif self.mode == EventBusMode.ASYNCHRONOUS:
                use_async = True
            else:  # HYBRID
                use_async = event.metadata.async_processing

        # 发布事件
        if use_async and self.running:
            # 异步处理 - 添加到队列
            try:
                # 使用优先级和时间戳作为排序键
                priority_key = (
                    -event.metadata.priority.value,  # 负值使高优先级排在前面
                    event.metadata.publication_time
                )
                self.event_queue.put((priority_key, event))
                logger.debug(
                    f"事件已加入异步队列: {event.event_type.name} (ID: {event.metadata.event_id})")
            except Exception as e:
                logger.error(f"将事件添加到队列时出错: {e}")
                self.dropped_events_count += 1
        else:
            # 同步处理 - 直接分发
            self._dispatch_event(event)

        return event.metadata.event_id

    def _worker_loop(self):
        """工作线程循环 - 处理队列中的事件"""
        logger.info(f"事件总线工作线程启动: {threading.current_thread().name}")

        while self.running:
            try:
                # 获取下一个事件（有超时以便检查running标志）
                try:
                    _, event = self.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 分发事件
                try:

                    self._dispatch_event(event)
                finally:
                    # 标记任务完成
                    self.event_queue.task_done()

            except Exception as e:
                logger.error(f"事件处理工作线程错误: {e}")
                time.sleep(0.1)  # 防止错误循环占用过多CPU

        logger.info(f"事件总线工作线程退出: {threading.current_thread().name}")

    def _dispatch_event(self, event: Event):
        """
        分发事件到所有匹配的处理器

        Args:
            event: 要分发的事件
        """
        dispatched = False
        event_type = event.event_type

        try:
            # 处理特定事件类型的订阅
            if event_type in self.subscriptions:
                for subscription in self.subscriptions[event_type]:
                    if subscription.filter.match(event):
                        self._deliver_to_handler(event, subscription)
                        dispatched = True

            # 处理全局订阅
            for subscription in self.global_subscriptions:
                if subscription.filter.match(event):
                    self._deliver_to_handler(event, subscription)
                    dispatched = True

            # 更新统计
            if dispatched:
                self.delivered_events_count += 1
                # 设置事件送达时间
                event.metadata.delivery_time = time.time()
            else:
                self.dropped_events_count += 1
                logger.debug(
                    f"事件未送达任何处理器: {event.event_type.name} (ID: {event.metadata.event_id})")

        except Exception as e:
            logger.error(f"分发事件时出错: {e}, 事件: {event.event_type.name}")
            self.dropped_events_count += 1

    def _deliver_to_handler(self, event: Event, subscription: SubscriptionInfo):
        """
        将事件送达到特定处理器

        Args:
            event: 要送达的事件
            subscription: 订阅信息
        """
        try:
            # 计时开始
            start_time = time.time()

            # 调用处理器
            subscription.handler(event)

            # 计算处理时间
            handling_time = time.time() - start_time

            # 更新订阅统计
            subscription.update_stats(handling_time)

            # 记录潜在的性能问题
            if handling_time > 0.1:  # 超过100ms视为潜在问题
                logger.warning(
                    f"事件处理器耗时较长: {handling_time:.3f}s, "
                    f"订阅者: {subscription.subscriber_id}, "
                    f"事件: {event.event_type.name}"
                )

        except Exception as e:
            logger.error(
                f"事件处理器错误: {e}, "
                f"订阅者: {subscription.subscriber_id}, "
                f"事件: {event.event_type.name}"
            )

    def _add_to_history(self, event: Event):
        """添加事件到历史记录"""
        with self.lock:
            self.event_history.append(event)
            # 限制历史记录大小
            while len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取事件总线统计信息

        Returns:
            Dict: 包含统计信息的字典
        """
        with self.lock:
            stats = {
                'published_count': self.published_events_count,
                'delivered_count': self.delivered_events_count,
                'dropped_count': self.dropped_events_count,
                'subscription_count': sum(
                    len(subs) for subs in self.subscriptions.values()) + len(
                    self.global_subscriptions),
                'queue_size': self.event_queue.qsize() if self.running else 0,
                'mode': self.mode.value,
                'worker_count': len(self.workers),
                'active': self.running
            }

            # 添加事件类型分布
            type_distribution = {}
            for event in self.event_history:
                event_type = event.event_type.name
                if event_type in type_distribution:
                    type_distribution[event_type] += 1
                else:
                    type_distribution[event_type] = 1

            stats['event_type_distribution'] = type_distribution

            return stats

    def get_subscriber_stats(self, subscriber_id: str) -> Optional[
        Dict[str, Any]]:
        """
        获取特定订阅者的统计信息

        Args:
            subscriber_id: 订阅者ID

        Returns:
            Dict or None: 包含订阅者统计信息的字典，如果订阅者不存在则返回None
        """
        with self.lock:
            if subscriber_id not in self.subscriber_map:
                return None

            stats = {
                'subscriber_id': subscriber_id,
                'subscription_count': len(self.subscriber_map[subscriber_id]),
                'subscriptions': {}
            }

            # 收集每个订阅的统计信息
            for event_type, subscription in self.subscriber_map[
                subscriber_id].items():
                event_type_name = 'GLOBAL' if event_type is None else event_type.name
                stats['subscriptions'][event_type_name] = {
                    'handled_count': subscription.handled_count,
                    'last_handled_time': subscription.last_handled_time,
                    'avg_handling_time': (
                        subscription.total_handling_time / subscription.handled_count
                        if subscription.handled_count > 0 else 0
                    )
                }

            return stats

    def clear_history(self):
        """清除事件历史记录"""
        with self.lock:
            self.event_history.clear()
            logger.debug("事件历史已清除")

    def get_event_history(self) -> List[Event]:
        """
        获取事件历史记录

        Returns:
            List[Event]: 事件历史记录列表（副本）
        """
        with self.lock:
            return list(self.event_history)  # 返回副本

    def set_max_history_size(self, size: int):
        """
        设置事件历史记录最大容量

        Args:
            size: 最大历史记录数量
        """
        with self.lock:
            self.max_history_size = max(10, size)  # 至少保留10条
            # 如果当前历史记录超过新的最大容量，进行裁剪
            while len(self.event_history) > self.max_history_size:
                self.event_history.pop(0)

            logger.debug(f"事件历史容量已设置为 {self.max_history_size}")

    def create_and_publish(self,
                           event_type: EventType,
                           data: Any = None,
                           source_id: str = "system",
                           priority: EventPriority = EventPriority.NORMAL,
                           async_processing: Optional[bool] = None) -> str:
        """
        创建并发布事件的便捷方法

        Args:
            event_type: 事件类型
            data: 事件数据
            source_id: 事件源ID
            priority: 事件优先级
            async_processing: 是否异步处理

        Returns:
            str: 事件ID
        """
        # 创建事件元数据
        metadata = EventMetadata(
            source_id=source_id,
            priority=priority,
            async_processing=async_processing if async_processing is not None else True
        )

        # 创建事件
        event = Event(
            event_type=event_type,
            data=data,
            metadata=metadata
        )

        # 发布事件
        return self.publish(event)

    def wait_for_delivery(self, timeout: Optional[float] = None) -> bool:
        """
        等待所有异步事件被送达

        Args:
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            bool: 是否所有事件都已送达
        """
        try:
            self.event_queue.join()
            return True
        except (KeyboardInterrupt, Exception):
            return False


# 便捷函数 - 获取事件总线单例
def get_event_bus() -> EventBus:
    """获取事件总线单例实例"""
    return EventBus.get_instance()
