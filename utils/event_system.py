# -*- coding: utf-8 -*-
"""
事件系统模块 - 提供基于发布/订阅模式的事件系统

本模块提供:
1. 基于观察者模式的事件系统
2. 支持事件过滤和优先级
3. 线程安全的事件分发
4. 事件历史记录和回放功能
"""
import time
import threading
import logging
import queue
from collections import defaultdict
from utils.logger_config import setup_logger
from utils.event_serializer import serialize_event, deserialize_event

# 获取日志记录器
logger = setup_logger("EventSystem")


# 明确定义EventSystem类
class EventSystem:
    """
    基本事件系统 - 提供发布/订阅功能

    支持事件订阅、发布和基本的事件历史记录
    """

    def __init__(self, history_capacity=100, enable_logging=True):
        """
        初始化事件系统

        Args:
            history_capacity: 事件历史记录容量
            enable_logging: 是否启用事件日志
        """
        # 订阅字典：事件类型 -> 订阅者列表，每个订阅者包含处理器、过滤器和优先级
        self.subscribers = defaultdict(list)

        # 锁，保证线程安全
        self.lock = threading.RLock()

        # 事件历史
        self.event_history = []
        self.history_capacity = history_capacity

        # 事件统计
        self.event_stats = defaultdict(int)

        # 日志设置
        self.enable_logging = enable_logging
        # 设置logger
        if enable_logging:
            self.logger = logging.getLogger("EventSystem")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            # 创建无操作的logger
            self.logger = logging.getLogger("NullLogger")
            self.logger.addHandler(logging.NullHandler())

        if enable_logging:
            self.logger.info(
                f"事件系统初始化: 历史容量={history_capacity}, 日志={enable_logging}")


    # 在基本的 EventSystem 类中也添加这些方法，或者可以让它继承自 EnhancedEventSystem

    def save_event_history(self, filepath, event_type=None, limit=None):
        """
        将事件历史保存到文件

        Args:
            filepath: 保存的文件路径
            event_type: 可选，过滤特定类型的事件
            limit: 可选，限制保存的事件数量

        Returns:
            int: 保存的事件数量
        """
        # 简单实现，或者复用 EnhancedEventSystem 的方法
        try:
            from utils.event_serializer import serialize_event
            import json
            import os

            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            events_to_save = self.get_event_history(event_type, limit)

            serialized_events = []
            for event_type, event_data in events_to_save:
                serialized_event = serialize_event(event_type, event_data)
                serialized_events.append(serialized_event)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_events, f, indent=2)

            if self.enable_logging:
                self.logger.info(
                    f"已保存 {len(serialized_events)} 个事件到 {filepath}")

            return len(serialized_events)
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"保存事件历史失败: {e}")
            return 0

    def load_event_history(self, filepath):
        """
        从文件加载事件历史

        Args:
            filepath: 事件历史文件路径

        Returns:
            list: 加载的事件列表，每项为(event_type, event_data)
        """
        # 简单实现，或者复用 EnhancedEventSystem 的方法
        try:
            from utils.event_serializer import deserialize_event
            import json
            import os

            if not os.path.exists(filepath):
                if self.enable_logging:
                    self.logger.warning(f"事件历史文件不存在: {filepath}")
                return []

            with open(filepath, 'r', encoding='utf-8') as f:
                serialized_events = json.load(f)

            loaded_events = []
            for serialized_event in serialized_events:
                event_type, event_data = deserialize_event(serialized_event)
                loaded_events.append((event_type, event_data))

            if self.enable_logging:
                self.logger.info(
                    f"已从 {filepath} 加载 {len(loaded_events)} 个事件")

            return loaded_events
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"加载事件历史失败: {e}")
            return []

    def subscribe(self, event_type, handler):
        """
        订阅特定类型的事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数，接收事件数据作为参数

        Returns:
            bool: 是否成功订阅
        """
        return self.subscribe_with_priority(event_type, handler)

    def subscribe_with_filter(self, event_type, handler, filter_func=None):
        """
        使用过滤器订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数，接收事件数据作为参数
            filter_func: 过滤函数，接收事件数据并返回布尔值

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
            handler: 事件处理函数
            priority: 优先级，数字越大优先级越高
            filter_func: 可选的过滤函数

        Returns:
            bool: 是否成功订阅
        """
        with self.lock:
            # 检查是否已订阅
            for sub in self.subscribers[event_type]:
                if sub['handler'] == handler:
                    # 更新现有订阅的优先级和过滤器
                    sub['priority'] = priority
                    sub['filter'] = filter_func

                    if self.enable_logging:
                        logger.info(
                            f"更新处理器: {event_type}, 优先级: {priority}")

                    # 重新排序订阅者列表
                    self._sort_subscribers(event_type)
                    return True

            # 创建包含处理器、过滤器和优先级的字典
            subscription = {
                'handler': handler,
                'filter': filter_func,
                'priority': priority
            }

            # 添加新订阅
            self.subscribers[event_type].append(subscription)

            # 按优先级排序
            self._sort_subscribers(event_type)

            if self.enable_logging:
                priority_msg = f"优先级: {priority}" if priority != 0 else ""
                filter_msg = "带过滤器" if filter_func else ""
                msg_parts = [part for part in [priority_msg, filter_msg] if
                             part]
                extra_info = f" ({', '.join(msg_parts)})" if msg_parts else ""

                logger.info(f"处理器订阅事件: {event_type}{extra_info}")

            return True

    def _sort_subscribers(self, event_type):
        """
        按优先级对订阅者排序

        Args:
            event_type: 事件类型
        """
        # 按优先级降序排序，优先级高的先执行
        self.subscribers[event_type].sort(
            key=lambda sub: sub.get('priority', 0),
            reverse=True
        )

    def unsubscribe(self, event_type, handler):
        """
        取消订阅特定类型的事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数

        Returns:
            bool: 是否成功取消订阅
        """
        with self.lock:
            if event_type not in self.subscribers:
                return False

            # 找到对应的订阅者
            for i, subscription in enumerate(self.subscribers[event_type]):
                if subscription['handler'] == handler:
                    # 移除订阅
                    self.subscribers[event_type].pop(i)

                    if self.enable_logging:
                        logger.info(f"处理器取消订阅事件: {event_type}")

                    return True

            return False

    def publish(self, event_type, data=None):
        """
        发布事件

        Args:
            event_type: 事件类型
            data: 事件数据，默认为None

        Returns:
            bool: 是否成功发布
        """
        # 确保数据始终是字典类型
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            data = {'value': data}

        # 添加时间戳（如果未指定）
        if 'timestamp' not in data:
            data['timestamp'] = time.time()

        # 添加事件类型
        data['event_type'] = event_type

        # 记录事件历史
        self._record_event(event_type, data)

        # 更新事件统计
        with self.lock:
            self.event_stats[event_type] += 1

        # 记录日志
        if self.enable_logging:
            logger.debug(f"发布事件: {event_type}")

        # 处理事件
        self._process_event(event_type, data)
        return True

    def _process_event(self, event_type, data):
        """处理单个事件"""
        with self.lock:
            # 获取事件订阅者列表的副本，避免在迭代过程中修改
            if event_type not in self.subscribers:
                return

            subscriptions = list(self.subscribers[event_type])

        # 调用每个订阅者的处理函数，按优先级顺序
        for subscription in subscriptions:
            try:
                handler = subscription['handler']
                filter_func = subscription.get('filter')

                # 如果有过滤器，先检查事件是否符合过滤条件
                if filter_func is not None:
                    try:
                        if not filter_func(data):
                            continue  # 不符合过滤条件，跳过此处理器
                    except Exception as filter_error:
                        logger.error(
                            f"事件过滤器错误 ({event_type}): {filter_error}")
                        continue  # 过滤器出错，跳过此处理器

                # 执行事件处理
                handler(data)
            except Exception as e:
                logger.error(f"事件处理器错误 ({event_type}): {e}")

    def _record_event(self, event_type, data):
        """记录事件到历史记录"""
        with self.lock:
            # 添加事件到历史
            self.event_history.append((event_type, data))

            # 限制历史记录大小
            if len(self.event_history) > self.history_capacity:
                self.event_history = self.event_history[-self.history_capacity:]

    def get_event_history(self, event_type=None, limit=None):
        """
        获取事件历史记录

        Args:
            event_type: 可选，事件类型过滤
            limit: 可选，限制返回的事件数量

        Returns:
            list: 事件历史记录
        """
        with self.lock:
            if event_type:
                # 过滤特定类型的事件
                filtered = [(t, d) for t, d in self.event_history if
                            t == event_type]
            else:
                # 所有事件
                filtered = list(self.event_history)

            # 限制数量
            if limit:
                filtered = filtered[-limit:]

            return filtered

    def get_event_stats(self):
        """
        获取事件统计信息

        Returns:
            dict: 事件统计字典，键为事件类型，值为事件数量
        """
        with self.lock:
            return dict(self.event_stats)

    def clear_history(self):
        """
        清除事件历史记录

        Returns:
            int: 清除的事件数量
        """
        with self.lock:
            count = len(self.event_history)
            self.event_history = []
            return count

    def reset_stats(self):
        """
        重置事件统计

        Returns:
            int: 重置的事件类型数量
        """
        with self.lock:
            count = len(self.event_stats)
            self.event_stats = defaultdict(int)
            return count


class EnhancedEventSystem(EventSystem):
    """
    增强版事件系统 - 提供高级事件处理能力

    特性:
    1. 线程安全的事件订阅和发布
    2. 支持事件过滤和优先级
    3. 事件历史记录和统计
    4. 异步事件处理和批处理
    """

    def __init__(self, history_capacity=100, enable_logging=True,
                 async_mode=False):
        """
        初始化增强版事件系统

        Args:
            history_capacity: 事件历史记录容量
            enable_logging: 是否启用事件日志
            async_mode: 是否启用异步事件处理
        """
        # 调用基类初始化
        super().__init__(history_capacity, enable_logging)

        # 异步模式设置
        self.async_mode = async_mode
        self.event_queue = queue.Queue() if async_mode else None
        self.async_thread = None
        self.async_running = False

        if async_mode:
            self._start_async_thread()

        if enable_logging:
            self.logger.info(f"增强事件系统初始化: 异步模式={async_mode}")

    # 在 EnhancedEventSystem 类中添加这些新方法

    def save_event_history(self, filepath, event_type=None, limit=None):
        """
        将事件历史保存到文件

        Args:
            filepath: 保存的文件路径
            event_type: 可选，过滤特定类型的事件
            limit: 可选，限制保存的事件数量

        Returns:
            int: 保存的事件数量
        """
        try:
            import json
            import os

            # 确保目录存在
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            # 获取要保存的事件历史
            events_to_save = self.get_event_history(event_type, limit)

            # 序列化事件
            serialized_events = []
            for event_type, event_data in events_to_save:
                serialized_event = serialize_event(event_type, event_data)
                serialized_events.append(serialized_event)

            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_events, f, indent=2)

            self.logger.info(
                f"已保存 {len(serialized_events)} 个事件到 {filepath}")
            return len(serialized_events)
        except Exception as e:
            self.logger.error(f"保存事件历史失败: {e}")
            return 0

    def load_event_history(self, filepath):
        """
        从文件加载事件历史

        Args:
            filepath: 事件历史文件路径

        Returns:
            list: 加载的事件列表，每项为(event_type, event_data)
        """
        try:
            import json
            import os

            if not os.path.exists(filepath):
                self.logger.warning(f"事件历史文件不存在: {filepath}")
                return []

            # 从文件加载
            with open(filepath, 'r', encoding='utf-8') as f:
                serialized_events = json.load(f)

            # 反序列化事件
            loaded_events = []
            for serialized_event in serialized_events:
                event_type, event_data = deserialize_event(serialized_event)
                loaded_events.append((event_type, event_data))

            self.logger.info(
                f"已从 {filepath} 加载 {len(loaded_events)} 个事件")
            return loaded_events
        except Exception as e:
            self.logger.error(f"加载事件历史失败: {e}")
            return []

    def _start_async_thread(self):
        """启动异步事件处理线程"""
        if self.async_thread is not None:
            return

        self.async_running = True
        self.async_thread = threading.Thread(target=self._process_event_queue)
        self.async_thread.daemon = True
        self.async_thread.start()
        logger.info("异步事件处理线程已启动")

    def _process_event_queue(self):
        """异步事件处理线程主循环"""
        while self.async_running:
            try:
                # 获取下一个事件
                event_type, data = self.event_queue.get(timeout=0.1)

                # 处理事件
                super()._process_event(event_type, data)

                # 标记任务完成
                self.event_queue.task_done()
            except queue.Empty:
                # 队列为空，继续等待
                pass
            except Exception as e:
                logger.error(f"处理事件队列时出错: {e}")
                time.sleep(0.01)  # 避免CPU过度占用

    def publish(self, event_type, data=None):
        """
        发布事件

        Args:
            event_type: 事件类型
            data: 事件数据，默认为None

        Returns:
            bool: 是否成功发布
        """
        # 确保数据始终是字典类型
        if data is None:
            data = {}
        elif not isinstance(data, dict):
            data = {'value': data}

        # 添加时间戳（如果未指定）
        if 'timestamp' not in data:
            data['timestamp'] = time.time()

        # 添加事件类型
        data['event_type'] = event_type

        # 记录事件历史
        self._record_event(event_type, data)

        # 更新事件统计
        with self.lock:
            self.event_stats[event_type] += 1

        # 记录日志
        if self.enable_logging:
            logger.debug(f"发布事件: {event_type}")

        # 异步或同步处理
        if self.async_mode:
            # 放入队列异步处理
            try:
                self.event_queue.put((event_type, data))
                return True
            except Exception as e:
                logger.error(f"将事件放入队列时出错: {e}")
                return False
        else:
            # 同步处理
            super()._process_event(event_type, data)
            return True

    def shutdown(self):
        """停止事件系统"""
        if self.async_mode and self.async_thread:
            self.async_running = False
            self.async_thread.join(timeout=1.0)
            self.async_thread = None
            logger.info("异步事件处理线程已关闭")


# 单例模式，提供全局事件系统实例
_event_system = None


def get_event_system(history_capacity=100, enable_logging=True,
                     async_mode=False):
    """
    获取全局事件系统实例

    Args:
        history_capacity: 事件历史记录容量
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


if __name__ == "__main__":
    # 示例代码：演示带优先级和过滤器的事件订阅
    print("测试事件系统的优先级和过滤功能...")

    # 创建事件系统
    events = EventSystem()


    # 定义不同优先级的事件处理器
    def low_priority_handler(data):
        print(f"低优先级处理器: {data}")


    def medium_priority_handler(data):
        print(f"中优先级处理器: {data}")


    def high_priority_handler(data):
        print(f"高优先级处理器: {data}")


    def critical_handler(data):
        print(f"紧急处理器: {data}")


    # 带不同优先级的订阅
    events.subscribe_with_priority("system_event", low_priority_handler,
                                   0)  # 最低优先级
    events.subscribe_with_priority("system_event", medium_priority_handler,
                                   5)  # 中等优先级
    events.subscribe_with_priority("system_event", high_priority_handler,
                                   10)  # 高优先级

    # 带优先级和过滤器的订阅
    events.subscribe_with_priority(
        "system_event",
        critical_handler,
        priority=100,  # 最高优先级
        filter_func=lambda data: data.get('status') == 'critical'  # 只处理紧急事件
    )

    # 发布事件
    print("\n发布普通事件:")
    events.publish("system_event", {"message": "系统正常运行", "importance": 3})

    print("\n发布紧急事件:")
    events.publish("system_event", {"message": "系统崩溃", "importance": 10,
                                    "status": "critical"})
