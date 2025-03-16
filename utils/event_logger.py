# utils/event_logger.py

import os
import json
import time
import logging
import threading
import gzip
import pickle
from datetime import datetime
from collections import defaultdict
import re


class EventLogger:
    """
    事件日志记录器 - 用于记录事件流和持久化事件

    这个类提供自动事件日志记录和定期事件持久化功能，
    它会监听所有事件，并定期将事件保存到文件中。
    增强功能包括：
    - 结构化事件分析
    - 事件过滤和分类
    - 压缩存储
    - 增量记录
    """

    def __init__(self, event_system, log_dir="logs/events",
                 auto_save_interval=300, max_events_per_file=1000,
                 use_compression=False, enable_analysis=True):
        """
        初始化事件日志记录器

        Args:
            event_system: 事件系统实例
            log_dir: 事件日志保存目录
            auto_save_interval: 自动保存间隔(秒)
            max_events_per_file: 每个文件最大事件数量
            use_compression: 是否使用压缩存储
            enable_analysis: 是否启用实时分析
        """
        self.event_system = event_system
        self.log_dir = log_dir
        self.auto_save_interval = auto_save_interval
        self.max_events_per_file = max_events_per_file
        self.use_compression = use_compression
        self.enable_analysis = enable_analysis

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

        # 增强: 事件分析和过滤
        self.event_patterns = {}  # 模式名称 -> 匹配规则
        self.event_categories = {}  # 类别名称 -> 事件类型列表
        self.event_stats = defaultdict(int)  # 事件类型 -> 数量
        self.sequence_patterns = []  # 序列模式列表
        self.active_sequences = []  # 活跃的序列跟踪

        # 增强: 增量记录
        self.incremental_mode = False
        self.current_log_filepath = None

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
        """记录事件到当前集合，应用过滤和分类"""
        with self.lock:
            # 应用过滤条件
            if not self._should_record_event(event_type, data):
                return

            # 更新事件统计
            self.event_stats[event_type] += 1

            # 如果启用了分析，更新序列模式
            if self.enable_analysis:
                self._update_sequence_tracking(event_type, data)

            # 将事件添加到当前集合
            self.current_events.append((event_type, data))

            # 检查是否需要自动保存
            current_time = time.time()

            # 根据记录模式决定保存方式
            if self.incremental_mode:
                # 增量模式：到达最大事件数量时追加
                if len(self.current_events) >= self.max_events_per_file:
                    self.append_events()
            else:
                # 标准模式：按间隔或数量保存
                if (
                        current_time - self.last_save_time >= self.auto_save_interval or
                        len(self.current_events) >= self.max_events_per_file):
                    self.save_current_events()

    def _should_record_event(self, event_type, data):
        """
        根据过滤条件决定是否记录事件

        Returns:
            bool: 是否应该记录该事件
        """
        # 检查是否有针对该事件类型的过滤规则
        if hasattr(self, 'event_filters') and event_type in self.event_filters:
            filter_rule = self.event_filters[event_type]

            # 如果过滤规则是函数，调用它
            if callable(filter_rule):
                return filter_rule(data)
            # 如果过滤规则是布尔值，直接返回
            elif isinstance(filter_rule, bool):
                return filter_rule

        # 默认记录所有事件
        return True

    def save_current_events(self):
        """保存当前收集的事件到文件"""
        with self.lock:
            if not self.current_events:
                return 0

            try:
                # 生成文件名: events_YYYYMMDD_HHMMSS.json[.gz]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"events_{timestamp}.json"
                if self.use_compression:
                    filename += ".gz"
                filepath = os.path.join(self.log_dir, filename)

                # 序列化事件
                from utils.event_serializer import serialize_event
                serialized_events = [
                    serialize_event(event_type, event_data)
                    for event_type, event_data in self.current_events
                ]

                # 根据是否压缩选择写入方式
                if self.use_compression:
                    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                        json.dump(serialized_events, f, ensure_ascii=False,
                                  indent=2)
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(serialized_events, f, ensure_ascii=False,
                                  indent=2)

                event_count = len(self.current_events)
                self.logger.info(f"已保存 {event_count} 个事件到 {filepath}")

                # 清空当前事件集
                self.current_events = []
                self.last_save_time = time.time()

                return event_count
            except Exception as e:
                self.logger.error(f"保存事件失败: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                return 0

    def enable_incremental_logging(self, jsonl_format=True):
        """
        启用增量记录模式

        Args:
            jsonl_format: 是否使用JSONL格式（每行一个事件）
        """
        self.incremental_mode = True
        self.jsonl_format = jsonl_format

        # 创建新的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_filename = f"events_incremental_{timestamp}"

        if self.jsonl_format:
            self.current_log_filename += ".jsonl"
        else:
            self.current_log_filename += ".json"

        if self.use_compression:
            self.current_log_filename += ".gz"

        self.current_log_filepath = os.path.join(self.log_dir,
                                                 self.current_log_filename)

        # 如果使用JSON格式，初始化文件
        if not self.jsonl_format:
            try:
                # 创建一个包含空数组的JSON文件
                if self.use_compression:
                    with gzip.open(self.current_log_filepath, 'wt',
                                   encoding='utf-8') as f:
                        f.write("[]")
                else:
                    with open(self.current_log_filepath, 'w',
                              encoding='utf-8') as f:
                        f.write("[]")
            except Exception as e:
                self.logger.error(f"初始化增量日志文件失败: {e}")
                self.incremental_mode = False

        self.logger.info(
            f"已启用增量记录模式，日志文件: {self.current_log_filepath}")

    def append_events(self):
        """将当前事件增量追加到日志文件"""
        with self.lock:
            if not self.current_events or not hasattr(self,
                                                      'incremental_mode') or not self.incremental_mode:
                return 0

            try:
                # 序列化事件
                from utils.event_serializer import serialize_event
                serialized_events = [
                    serialize_event(event_type, event_data)
                    for event_type, event_data in self.current_events
                ]

                # 根据格式和压缩选项选择写入方式
                if self.jsonl_format:
                    # JSONL格式：每行一个事件
                    if self.use_compression:
                        with gzip.open(self.current_log_filepath, 'at',
                                       encoding='utf-8') as f:
                            for event in serialized_events:
                                f.write(json.dumps(event,
                                                   ensure_ascii=False) + '\n')
                    else:
                        with open(self.current_log_filepath, 'a',
                                  encoding='utf-8') as f:
                            for event in serialized_events:
                                f.write(json.dumps(event,
                                                   ensure_ascii=False) + '\n')
                else:
                    # JSON格式：需要读取、修改和重写整个文件
                    # 注意：这种方式效率较低，不推荐用于频繁追加
                    if self.use_compression:
                        with gzip.open(self.current_log_filepath, 'rt',
                                       encoding='utf-8') as f:
                            events = json.load(f)
                        events.extend(serialized_events)
                        with gzip.open(self.current_log_filepath, 'wt',
                                       encoding='utf-8') as f:
                            json.dump(events, f, ensure_ascii=False)
                    else:
                        with open(self.current_log_filepath, 'r',
                                  encoding='utf-8') as f:
                            events = json.load(f)
                        events.extend(serialized_events)
                        with open(self.current_log_filepath, 'w',
                                  encoding='utf-8') as f:
                            json.dump(events, f, ensure_ascii=False)

                event_count = len(self.current_events)
                self.logger.info(
                    f"已增量追加 {event_count} 个事件到 {self.current_log_filepath}")

                # 清空当前事件集
                self.current_events = []
                self.last_save_time = time.time()

                return event_count
            except Exception as e:
                self.logger.error(f"增量追加事件失败: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                return 0

    def set_event_filters(self, filters):
        """
        设置事件过滤规则

        Args:
            filters: 过滤规则字典 {事件类型: 布尔值或过滤函数}
        """
        self.event_filters = filters
        self.logger.info(f"已设置事件过滤规则: {len(filters)} 条规则")

    def add_event_filter(self, event_type, filter_rule):
        """
        添加单个事件过滤规则

        Args:
            event_type: 事件类型
            filter_rule: 布尔值或过滤函数
        """
        if not hasattr(self, 'event_filters'):
            self.event_filters = {}

        self.event_filters[event_type] = filter_rule
        self.logger.info(f"已添加事件过滤规则: {event_type}")

    def add_event_pattern(self, pattern_name, event_type_pattern,
                          data_conditions=None):
        """
        添加事件模式

        Args:
            pattern_name: 模式名称
            event_type_pattern: 事件类型的正则表达式或字符串
            data_conditions: 事件数据的条件字典或函数
        """
        # 如果是字符串，转换为正则表达式
        if isinstance(event_type_pattern, str):
            event_type_pattern = re.compile(event_type_pattern)

        self.event_patterns[pattern_name] = (
        event_type_pattern, data_conditions)
        self.logger.info(f"已添加事件模式: {pattern_name}")

    def categorize_events(self, category_name, event_types):
        """
        将事件类型归类

        Args:
            category_name: 类别名称
            event_types: 事件类型列表
        """
        self.event_categories[category_name] = event_types
        self.logger.info(
            f"已添加事件类别: {category_name} ({len(event_types)} 种事件)")

    def add_sequence_pattern(self, name, event_sequence, timeout=None,
                             callback=None):
        """
        添加事件序列模式，用于检测事件序列

        Args:
            name: 模式名称
            event_sequence: 事件类型序列列表
            timeout: 序列超时时间(秒)
            callback: 检测到序列时的回调函数
        """
        self.sequence_patterns.append({
            'name': name,
            'sequence': event_sequence,
            'timeout': timeout,
            'callback': callback
        })
        self.logger.info(f"已添加事件序列模式: {name}")

    def _update_sequence_tracking(self, event_type, data):
        """更新序列跟踪状态"""
        # 检查是否可以启动新的序列
        for pattern in self.sequence_patterns:
            if pattern['sequence'][0] == event_type:
                # 创建新的序列跟踪
                self.active_sequences.append({
                    'pattern': pattern,
                    'position': 0,
                    'start_time': time.time(),
                    'events': []
                })

        # 更新现有序列
        completed_sequences = []
        for i, sequence in enumerate(self.active_sequences):
            pattern = sequence['pattern']
            position = sequence['position']

            # 检查当前事件是否匹配序列中的下一个事件
            if pattern['sequence'][position] == event_type:
                # 更新位置和事件历史
                sequence['position'] += 1
                sequence['events'].append((event_type, data))

                # 检查序列是否完成
                if sequence['position'] >= len(pattern['sequence']):
                    # 序列完成
                    if pattern['callback']:
                        try:
                            pattern['callback'](sequence['events'])
                        except Exception as e:
                            self.logger.error(f"执行序列回调失败: {e}")

                    self.logger.info(f"检测到事件序列: {pattern['name']}")
                    completed_sequences.append(i)

            # 检查超时
            elif pattern['timeout']:
                if time.time() - sequence['start_time'] > pattern['timeout']:
                    completed_sequences.append(i)

        # 移除完成或超时的序列（从后向前移除，避免索引问题）
        for i in sorted(completed_sequences, reverse=True):
            self.active_sequences.pop(i)

    def get_event_stats(self):
        """
        获取事件统计信息

        Returns:
            dict: 事件统计字典
        """
        return dict(self.event_stats)

    def get_categorized_stats(self):
        """
        获取分类事件统计

        Returns:
            dict: 类别 -> 事件统计字典
        """
        result = {}

        # 统计每个类别的事件数量
        for category, event_types in self.event_categories.items():
            category_stats = {}
            category_total = 0

            for event_type in event_types:
                count = self.event_stats.get(event_type, 0)
                category_stats[event_type] = count
                category_total += count

            result[category] = {
                'total': category_total,
                'events': category_stats
            }

        return result

    def analyze_event_patterns(self, events=None):
        """
        分析事件模式

        Args:
            events: 要分析的事件列表，默认使用当前事件

        Returns:
            dict: 模式名称 -> 匹配事件列表
        """
        if events is None:
            events = self.current_events

        results = defaultdict(list)

        # 检查每个事件是否匹配任何模式
        for event_type, data in events:
            for pattern_name, (
            type_pattern, data_conditions) in self.event_patterns.items():
                # 检查事件类型是否匹配
                if isinstance(type_pattern, re.Pattern):
                    type_match = bool(type_pattern.match(event_type))
                else:
                    type_match = (event_type == type_pattern)

                if not type_match:
                    continue

                # 检查数据条件
                if data_conditions is None:
                    # 无数据条件，类型匹配即可
                    results[pattern_name].append((event_type, data))
                elif callable(data_conditions):
                    # 使用函数检查
                    if data_conditions(data):
                        results[pattern_name].append((event_type, data))
                elif isinstance(data_conditions, dict):
                    # 检查字典条件
                    match = True
                    for key, value in data_conditions.items():
                        if key not in data or data[key] != value:
                            match = False
                            break

                    if match:
                        results[pattern_name].append((event_type, data))

        return dict(results)

    def export_analysis(self, filepath=None):
        """
        导出事件分析报告

        Args:
            filepath: 输出文件路径，默认为自动生成

        Returns:
            str: 输出文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.log_dir,
                                    f"event_analysis_{timestamp}.json")

        try:
            # 生成分析报告
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_events': sum(self.event_stats.values()),
                'event_stats': self.get_event_stats(),
                'categorized_stats': self.get_categorized_stats(),
                'pattern_matches': self.analyze_event_patterns()
            }

            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            self.logger.info(f"事件分析报告已导出: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"导出分析报告失败: {e}")
            return None

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
        if self.incremental_mode:
            saved = self.append_events()
        else:
            saved = self.save_current_events()

        # 导出最终分析报告
        if self.enable_analysis:
            self.export_analysis()

        self.logger.info(f"事件日志记录器已停止，最后保存了 {saved} 个事件")
