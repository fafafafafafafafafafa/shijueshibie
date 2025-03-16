# -*- coding: utf-8 -*-
"""
组件性能指标收集模块 - 提供组件性能数据的采集、聚合和分析
支持多维度指标监控、异常检测和性能报告生成
"""

import logging
import threading
import time
import json
import os
import uuid
import statistics
import math
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field

# 导入需要的模块
from core.component_interface import ComponentInterface

logger = logging.getLogger("ComponentMetrics")


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"  # 计数器 - 只增不减的单调递增值
    GAUGE = "gauge"  # 仪表盘 - 可增可减的当前值
    TIMER = "timer"  # 计时器 - 测量持续时间
    HISTOGRAM = "histogram"  # 直方图 - 值分布统计
    METER = "meter"  # 流量计 - 测量事件频率


class MetricUnit(Enum):
    """指标单位枚举"""
    COUNT = "count"  # 计数
    MILLISECONDS = "ms"  # 毫秒
    SECONDS = "s"  # 秒
    BYTES = "bytes"  # 字节
    KILOBYTES = "KB"  # 千字节
    MEGABYTES = "MB"  # 兆字节
    PERCENT = "percent"  # 百分比
    OPERATIONS = "ops"  # 操作数
    EVENTS = "events"  # 事件数
    ERRORS = "errors"  # 错误数
    CUSTOM = "custom"  # 自定义单位


class AggregationMethod(Enum):
    """聚合方法枚举"""
    SUM = "sum"  # 求和
    AVERAGE = "avg"  # 平均值
    MIN = "min"  # 最小值
    MAX = "max"  # 最大值
    COUNT = "count"  # 计数
    PERCENTILE = "percentile"  # 百分位数
    RATE = "rate"  # 速率
    STDDEV = "stddev"  # 标准差


@dataclass
class MetricValue:
    """指标值记录"""
    value: float  # 指标值
    timestamp: float = field(default_factory=time.time)  # 记录时间戳
    tags: Dict[str, str] = field(default_factory=dict)  # 标签
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str  # 指标名称
    type: MetricType  # 指标类型
    unit: MetricUnit  # 指标单位
    description: str = ""  # 指标描述
    tags: Dict[str, str] = field(default_factory=dict)  # 默认标签
    threshold_warning: Optional[float] = None  # 警告阈值
    threshold_critical: Optional[float] = None  # 严重阈值
    aggregation_methods: List[AggregationMethod] = field(
        default_factory=list)  # 支持的聚合方法


class ComponentMetricsCollector:
    """
    组件性能指标收集器 - 收集单个组件的性能指标
    """

    def __init__(self, component_id: str, component_type: str):
        """
        初始化组件指标收集器

        Args:
            component_id: 组件ID
            component_type: 组件类型
        """
        self.component_id = component_id
        self.component_type = component_type

        # 指标定义 {metric_name: MetricDefinition}
        self.metric_definitions: Dict[str, MetricDefinition] = {}

        # 指标数据存储 {metric_name: List[MetricValue]}
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)

        # 最近值缓存 {metric_name: MetricValue}
        self.latest_values: Dict[str, MetricValue] = {}

        # 直方图数据 {metric_name: List[float]}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # 计数器当前值 {metric_name: float}
        self.counters: Dict[str, float] = defaultdict(float)

        # 计时器上下文管理器
        self.active_timers: Dict[str, float] = {}

        # 指标保留期限（秒）
        self.retention_period = 3600  # 1小时

        # 是否启用异常检测
        self.anomaly_detection_enabled = False

        # 异常检测配置
        self.anomaly_detection_config: Dict[str, Any] = {
            'z_score_threshold': 3.0,  # Z分数阈值
            'window_size': 30,  # 检测窗口大小
            'min_data_points': 10,  # 最小数据点数
        }

        # 检测到的异常 {metric_name: List[Dict]}
        self.detected_anomalies: Dict[str, List[Dict]] = defaultdict(list)

        # 线程锁
        self.lock = threading.RLock()

        logger.debug(
            f"已初始化组件指标收集器: {component_id} ({component_type})")

    def register_metric(self, definition: MetricDefinition) -> bool:
        """
        注册指标定义

        Args:
            definition: 指标定义

        Returns:
            bool: 是否成功注册
        """
        with self.lock:
            if definition.name in self.metric_definitions:
                logger.warning(f"指标 '{definition.name}' 已存在，将被覆盖")

            self.metric_definitions[definition.name] = definition

            # 初始化存储
            if definition.name not in self.metrics:
                self.metrics[definition.name] = []

            # 为直方图初始化数据结构
            if definition.type == MetricType.HISTOGRAM and definition.name not in self.histograms:
                self.histograms[definition.name] = []

            logger.debug(
                f"已注册指标: {definition.name}, 类型: {definition.type.value}")
            return True

    def record_value(self, metric_name: str, value: float,
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录指标值

        Args:
            metric_name: 指标名称
            value: 指标值
            tags: 标签
            metadata: 元数据

        Returns:
            bool: 是否成功记录
        """
        with self.lock:
            # 检查指标是否已注册
            if metric_name not in self.metric_definitions:
                logger.warning(f"尝试记录未注册的指标: {metric_name}")
                return False

            definition = self.metric_definitions[metric_name]

            # 合并标签
            combined_tags = dict(definition.tags)
            if tags:
                combined_tags.update(tags)

            # 创建指标值记录
            metric_value = MetricValue(
                value=value,
                timestamp=time.time(),
                tags=combined_tags,
                metadata=metadata or {}
            )

            # 根据指标类型处理
            if definition.type == MetricType.COUNTER:
                # 计数器 - 累加值
                if metric_name not in self.counters:
                    self.counters[metric_name] = value
                else:
                    # 确保计数器只增不减
                    if value >= 0:
                        self.counters[metric_name] += value
                    else:
                        logger.warning(f"计数器不能减少: {metric_name}")
                        return False

                # 更新实际记录的值为当前计数器值
                metric_value.value = self.counters[metric_name]

            elif definition.type == MetricType.GAUGE:
                # 仪表盘 - 直接使用当前值
                pass

            elif definition.type == MetricType.HISTOGRAM:
                # 直方图 - 记录值分布
                self.histograms[metric_name].append(value)

                # 限制直方图大小
                max_histogram_size = 1000
                if len(self.histograms[metric_name]) > max_histogram_size:
                    self.histograms[metric_name] = self.histograms[metric_name][
                                                   -max_histogram_size:]

            # 添加到指标数据
            self.metrics[metric_name].append(metric_value)

            # 更新最近值
            self.latest_values[metric_name] = metric_value

            # 检查阈值
            self._check_thresholds(metric_name, metric_value)

            # 如果启用了异常检测，检查异常
            if self.anomaly_detection_enabled:
                self._detect_anomalies(metric_name)

            # 清理过期数据
            self._cleanup_old_data(metric_name)

            return True

    def increment(self, metric_name: str, increment: float = 1.0,
                  tags: Optional[Dict[str, str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        增加计数器

        Args:
            metric_name: 计数器名称
            increment: 增加量
            tags: 标签
            metadata: 元数据

        Returns:
            bool: 是否成功增加
        """
        with self.lock:
            # 检查指标是否已注册
            if metric_name not in self.metric_definitions:
                logger.warning(f"尝试增加未注册的计数器: {metric_name}")
                return False

            definition = self.metric_definitions[metric_name]

            # 确保是计数器类型
            if definition.type != MetricType.COUNTER:
                logger.warning(f"指标 {metric_name} 不是计数器类型")
                return False

            # 记录新值
            return self.record_value(metric_name, increment, tags, metadata)

    def start_timer(self, metric_name: str) -> str:
        """
        启动计时器

        Args:
            metric_name: 计时器名称

        Returns:
            str: 计时器ID
        """
        with self.lock:
            # 检查指标是否已注册
            if metric_name not in self.metric_definitions:
                logger.warning(f"尝试启动未注册的计时器: {metric_name}")
                return ""

            definition = self.metric_definitions[metric_name]

            # 确保是计时器类型
            if definition.type != MetricType.TIMER:
                logger.warning(f"指标 {metric_name} 不是计时器类型")
                return ""

            # 生成计时器ID
            timer_id = f"{metric_name}_{uuid.uuid4().hex}"

            # 记录开始时间
            self.active_timers[timer_id] = time.time()

            return timer_id

    def stop_timer(self, timer_id: str, tags: Optional[Dict[str, str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[
        float]:
        """
        停止计时器并记录持续时间

        Args:
            timer_id: 计时器ID
            tags: 标签
            metadata: 元数据

        Returns:
            Optional[float]: 持续时间（毫秒），失败返回None
        """
        with self.lock:
            if timer_id not in self.active_timers:
                logger.warning(f"尝试停止不存在的计时器: {timer_id}")
                return None

            # 计算持续时间
            start_time = self.active_timers[timer_id]
            duration_ms = (time.time() - start_time) * 1000  # 转换为毫秒

            # 移除计时器
            del self.active_timers[timer_id]

            # 提取指标名称
            metric_name = timer_id.split('_')[0]

            # 记录持续时间
            self.record_value(metric_name, duration_ms, tags, metadata)

            return duration_ms

    def update_gauge(self, metric_name: str, value: float,
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新仪表盘指标

        Args:
            metric_name: 指标名称
            value: 当前值
            tags: 标签
            metadata: 元数据

        Returns:
            bool: 是否成功更新
        """
        with self.lock:
            # 检查指标是否已注册
            if metric_name not in self.metric_definitions:
                logger.warning(f"尝试更新未注册的仪表盘: {metric_name}")
                return False

            definition = self.metric_definitions[metric_name]

            # 确保是仪表盘类型
            if definition.type != MetricType.GAUGE:
                logger.warning(f"指标 {metric_name} 不是仪表盘类型")
                return False

            # 记录新值
            return self.record_value(metric_name, value, tags, metadata)

    def mark_event(self, metric_name: str, count: float = 1.0,
                   tags: Optional[Dict[str, str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        标记事件(Meter)

        Args:
            metric_name: 指标名称
            count: 事件数量
            tags: 标签
            metadata: 元数据

        Returns:
            bool: 是否成功标记
        """
        with self.lock:
            # 检查指标是否已注册
            if metric_name not in self.metric_definitions:
                logger.warning(f"尝试标记未注册的事件指标: {metric_name}")
                return False

            definition = self.metric_definitions[metric_name]

            # 确保是Meter类型
            if definition.type != MetricType.METER:
                logger.warning(f"指标 {metric_name} 不是Meter类型")
                return False

            # 记录事件
            return self.record_value(metric_name, count, tags, metadata)

    def get_latest_value(self, metric_name: str) -> Optional[MetricValue]:
        """
        获取指标最近值

        Args:
            metric_name: 指标名称

        Returns:
            Optional[MetricValue]: 最近的指标值，如果不存在则返回None
        """
        with self.lock:
            return self.latest_values.get(metric_name)

    def get_histogram_stats(self, metric_name: str) -> Dict[str, Any]:
        """
        获取直方图统计信息

        Args:
            metric_name: 指标名称

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        with self.lock:
            if metric_name not in self.histograms or not self.histograms[
                metric_name]:
                return {}

            values = self.histograms[metric_name]

            try:
                stats = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stddev': statistics.stdev(values) if len(
                        values) > 1 else 0,
                }

                # 计算百分位数
                stats['p75'] = self._percentile(values, 75)
                stats['p90'] = self._percentile(values, 90)
                stats['p95'] = self._percentile(values, 95)
                stats['p99'] = self._percentile(values, 99)

                return stats
            except Exception as e:
                logger.error(f"计算直方图统计信息出错: {e}")
                return {'error': str(e)}

    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        d0 = sorted_values[int(f)] * (c - k)
        d1 = sorted_values[int(c)] * (k - f)
        return d0 + d1

    def get_aggregated_value(self, metric_name: str, method: AggregationMethod,
                             time_window: Optional[float] = None,
                             tags_filter: Optional[Dict[str, str]] = None) -> \
    Optional[float]:
        """
        获取聚合值

        Args:
            metric_name: 指标名称
            method: 聚合方法
            time_window: 时间窗口（秒），None表示所有数据
            tags_filter: 标签过滤器

        Returns:
            Optional[float]: 聚合值，如果不存在则返回None
        """
        with self.lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None

            # 过滤数据
            filtered_values = self._filter_metrics(metric_name, time_window,
                                                   tags_filter)
            if not filtered_values:
                return None

            # 提取值列表
            values = [mv.value for mv in filtered_values]

            # 应用聚合方法
            try:
                if method == AggregationMethod.SUM:
                    return sum(values)
                elif method == AggregationMethod.AVERAGE:
                    return statistics.mean(values)
                elif method == AggregationMethod.MIN:
                    return min(values)
                elif method == AggregationMethod.MAX:
                    return max(values)
                elif method == AggregationMethod.COUNT:
                    return len(values)
                elif method == AggregationMethod.STDDEV:
                    return statistics.stdev(values) if len(values) > 1 else 0
                elif method == AggregationMethod.RATE:
                    # 计算单位时间内的事件率
                    if len(filtered_values) < 2:
                        return 0
                    time_span = filtered_values[-1].timestamp - filtered_values[
                        0].timestamp
                    if time_span <= 0:
                        return 0
                    return sum(values) / time_span
                elif method == AggregationMethod.PERCENTILE:
                    # 默认95百分位
                    return self._percentile(values, 95)
                else:
                    logger.warning(f"不支持的聚合方法: {method}")
                    return None
            except Exception as e:
                logger.error(f"计算聚合值出错: {e}")
                return None

    def _filter_metrics(self, metric_name: str,
                        time_window: Optional[float] = None,
                        tags_filter: Optional[Dict[str, str]] = None) -> List[
        MetricValue]:
        """过滤指标数据"""
        metrics = self.metrics[metric_name]

        # 应用时间窗口过滤
        if time_window is not None:
            now = time.time()
            metrics = [m for m in metrics if now - m.timestamp <= time_window]

        # 应用标签过滤
        if tags_filter:
            metrics = [m for m in metrics if
                       self._match_tags(m.tags, tags_filter)]

        return metrics

    def _match_tags(self, tags: Dict[str, str],
                    filter_tags: Dict[str, str]) -> bool:
        """检查标签是否匹配过滤条件"""
        for key, value in filter_tags.items():
            if key not in tags or tags[key] != value:
                return False
        return True

    def _check_thresholds(self, metric_name: str, metric_value: MetricValue):
        """检查指标值是否超过阈值"""
        definition = self.metric_definitions[metric_name]
        value = metric_value.value

        # 检查警告阈值
        if definition.threshold_warning is not None and value >= definition.threshold_warning:
            logger.warning(
                f"指标 {metric_name} 超过警告阈值: {value} >= {definition.threshold_warning}")

            # 添加阈值超过标记
            metric_value.metadata['threshold_exceeded'] = 'warning'

        # 检查严重阈值
        if definition.threshold_critical is not None and value >= definition.threshold_critical:
            logger.error(
                f"指标 {metric_name} 超过严重阈值: {value} >= {definition.threshold_critical}")

            # 添加阈值超过标记
            metric_value.metadata['threshold_exceeded'] = 'critical'

    def _detect_anomalies(self, metric_name: str):
        """
        检测指标异常

        使用Z分数方法检测异常值
        """
        if not self.anomaly_detection_enabled:
            return

        # 获取配置
        z_threshold = self.anomaly_detection_config['z_score_threshold']
        window_size = self.anomaly_detection_config['window_size']
        min_data_points = self.anomaly_detection_config['min_data_points']

        # 获取最近的数据点
        recent_values = self.metrics[metric_name][-window_size:]

        # 确保有足够的数据点
        if len(recent_values) < min_data_points:
            return

        # 提取值
        values = [mv.value for mv in recent_values]

        try:
            # 计算均值和标准差
            mean = statistics.mean(values[:-1])  # 排除最新的点
            stddev = statistics.stdev(values[:-1])

            # 防止除零
            if stddev == 0:
                return

            # 计算最新点的Z分数
            latest_value = values[-1]
            z_score = abs((latest_value - mean) / stddev)

            # 检查是否异常
            if z_score > z_threshold:
                # 记录异常
                anomaly = {
                    'metric_name': metric_name,
                    'timestamp': time.time(),
                    'value': latest_value,
                    'expected_range': (
                    mean - z_threshold * stddev, mean + z_threshold * stddev),
                    'z_score': z_score
                }

                self.detected_anomalies[metric_name].append(anomaly)

                # 记录到日志
                logger.warning(
                    f"检测到指标异常: {metric_name}, 值: {latest_value}, Z分数: {z_score:.2f}")

        except Exception as e:
            logger.error(f"异常检测出错: {e}")

    def _cleanup_old_data(self, metric_name: str):
        """清理过期数据"""
        if self.retention_period <= 0:
            return

        now = time.time()
        cutoff = now - self.retention_period

        # 清理指标数据
        self.metrics[metric_name] = [m for m in self.metrics[metric_name] if
                                     m.timestamp >= cutoff]

    def enable_anomaly_detection(self, enabled: bool = True,
                                 config: Optional[Dict[str, Any]] = None):
        """
        启用或禁用异常检测

        Args:
            enabled: 是否启用
            config: 异常检测配置
        """
        with self.lock:
            self.anomaly_detection_enabled = enabled

            if config:
                self.anomaly_detection_config.update(config)

            if enabled:
                logger.info(f"已启用组件指标异常检测: {self.component_id}")
            else:
                logger.info(f"已禁用组件指标异常检测: {self.component_id}")

    def set_retention_period(self, period_seconds: int):
        """
        设置指标数据保留期限

        Args:
            period_seconds: 保留期限（秒）
        """
        with self.lock:
            self.retention_period = period_seconds
            logger.debug(f"已设置指标保留期限: {period_seconds}秒")

    def get_anomalies(self, metric_name: Optional[str] = None,
                      limit: int = 100) -> Dict[str, List[Dict]]:
        """
        获取检测到的异常

        Args:
            metric_name: 指标名称，None表示所有指标
            limit: 每个指标的最大记录数

        Returns:
            Dict[str, List[Dict]]: 异常记录字典
        """
        with self.lock:
            if metric_name:
                if metric_name not in self.detected_anomalies:
                    return {metric_name: []}
                return {
                    metric_name: self.detected_anomalies[metric_name][-limit:]}
            else:
                return {name: anomalies[-limit:] for name, anomalies in
                        self.detected_anomalies.items()}

    def clear_anomalies(self, metric_name: Optional[str] = None):
        """
        清除检测到的异常

        Args:
            metric_name: 指标名称，None表示所有指标
        """
        with self.lock:
            if metric_name:
                if metric_name in self.detected_anomalies:
                    self.detected_anomalies[metric_name] = []
            else:
                self.detected_anomalies.clear()

    def export_metrics(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        导出指标数据

        Args:
            filepath: 输出文件路径，None表示自动生成

        Returns:
            Optional[str]: 输出文件路径，失败返回None
        """
        with self.lock:
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"metrics_{self.component_id}_{timestamp}.json"

            try:
                # 准备导出数据
                export_data = {
                    'component_id': self.component_id,
                    'component_type': self.component_type,
                    'timestamp': time.time(),
                    'metrics': {}
                }

                # 导出每个指标的数据
                for metric_name, definition in self.metric_definitions.items():
                    metric_data = {
                        'definition': {
                            'name': definition.name,
                            'type': definition.type.value,
                            'unit': definition.unit.value,
                            'description': definition.description
                        },
                        'values': []
                    }

                    # 根据指标类型添加特定数据
                    if definition.type == MetricType.COUNTER:
                        metric_data['current_value'] = self.counters.get(
                            metric_name, 0)
                    elif definition.type == MetricType.GAUGE:
                        if metric_name in self.latest_values:
                            metric_data['current_value'] = self.latest_values[
                                metric_name].value
                    elif definition.type == MetricType.HISTOGRAM:
                        metric_data['stats'] = self.get_histogram_stats(
                            metric_name)

                    # 添加最近的值记录
                    for value in self.metrics.get(metric_name, [])[
                                 -100:]:  # 最多导出100个点
                        metric_data['values'].append({
                            'value': value.value,
                            'timestamp': value.timestamp,
                            'tags': value.tags
                        })

                    export_data['metrics'][metric_name] = metric_data

                # 写入文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)

                logger.info(f"已导出组件指标数据: {filepath}")
                return filepath

            except Exception as e:
                logger.error(f"导出指标数据失败: {e}")
                return None

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要

        Returns:
            Dict[str, Any]: 指标摘要
        """
        with self.lock:
            summary = {
                'component_id': self.component_id,
                'component_type': self.component_type,
                'timestamp': time.time(),
                'metrics_count': len(self.metric_definitions),
                'metrics': {}
            }

            for metric_name, definition in self.metric_definitions.items():
                metric_summary = {
                    'type': definition.type.value,
                    'unit': definition.unit.value,
                    'description': definition.description,
                    'data_points': len(self.metrics.get(metric_name, [])),
                }

                # 添加当前值
                if metric_name in self.latest_values:
                    metric_summary['latest_value'] = self.latest_values[
                        metric_name].value
                    metric_summary['latest_time'] = self.latest_values[
                        metric_name].timestamp

                # 根据指标类型添加特定数据
                if definition.type == MetricType.COUNTER:
                    metric_summary['current_count'] = self.counters.get(
                        metric_name, 0)
                elif definition.type == MetricType.HISTOGRAM:
                    stats = self.get_histogram_stats(metric_name)
                    if stats:
                        metric_summary['stats'] = stats

                # 添加阈值信息
                if definition.threshold_warning is not None:
                    metric_summary[
                        'threshold_warning'] = definition.threshold_warning
                if definition.threshold_critical is not None:
                    metric_summary[
                        'threshold_critical'] = definition.threshold_critical

                # 添加异常信息
                if metric_name in self.detected_anomalies and \
                        self.detected_anomalies[metric_name]:
                    metric_summary['anomalies_count'] = len(
                        self.detected_anomalies[metric_name])
                    metric_summary['last_anomaly'] = \
                    self.detected_anomalies[metric_name][-1]

                summary['metrics'][metric_name] = metric_summary

            return summary


class ComponentMetricsManager:
    """
    组件性能指标管理器 - 管理所有组件的性能指标
    提供系统级指标聚合、报告和监控功能
    """

    _instance = None  # 单例实例

    @classmethod
    def get_instance(cls) -> 'ComponentMetricsManager':
        """获取ComponentMetricsManager单例实例"""
        if cls._instance is None:
            cls._instance = ComponentMetricsManager()
        return cls._instance

    def __init__(self):
        """初始化组件指标管理器"""
        if ComponentMetricsManager._instance is not None:
            logger.warning(
                "ComponentMetricsManager是单例类，请使用get_instance()获取实例")
            return

        # 组件收集器字典 {component_id: ComponentMetricsCollector}
        self.collectors: Dict[str, ComponentMetricsCollector] = {}

        # 全局指标 {metric_name: MetricDefinition}
        self.global_metrics: Dict[str, MetricDefinition] = {}

        # 指标值存储 {metric_name: List[MetricValue]}
        self.global_metric_values: Dict[str, List[MetricValue]] = defaultdict(
            list)

        # 组件类型分组 {component_type: [component_id]}
        self.component_types: Dict[str, List[str]] = defaultdict(list)

        # 指标标签索引 {tag_key: {tag_value: [component_id]}}
        self.tag_index: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list))

        # 报告生成配置
        self.report_config = {
            'enabled': False,
            'interval': 3600,  # 报告间隔（秒）
            'output_dir': 'metrics_reports',
            'format': 'json',
            'include_anomalies': True,
            'include_stats': True
        }

        # 报告生成线程
        self.report_thread = None
        self.running = False

        # 指标聚合缓存
        self.aggregation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = 60  # 缓存过期时间（秒）
        self.cache_timestamps: Dict[str, float] = {}

        # 线程锁
        self.lock = threading.RLock()

        logger.info("组件指标管理器已初始化")

    def register_collector(self, collector: ComponentMetricsCollector) -> bool:
        """
        注册组件指标收集器

        Args:
            collector: 组件指标收集器

        Returns:
            bool: 是否成功注册
        """
        with self.lock:
            component_id = collector.component_id
            component_type = collector.component_type

            if component_id in self.collectors:
                logger.warning(f"组件收集器已存在，将被覆盖: {component_id}")

            # 注册收集器
            self.collectors[component_id] = collector

            # 更新组件类型分组
            if component_id not in self.component_types[component_type]:
                self.component_types[component_type].append(component_id)

            # 添加组件标签到索引
            # 使用组件类型作为默认标签
            self._add_to_tag_index('component_type', component_type,
                                   component_id)

            logger.debug(
                f"已注册组件指标收集器: {component_id} ({component_type})")
            return True

    def unregister_collector(self, component_id: str) -> bool:
        """
        取消注册组件指标收集器

        Args:
            component_id: 组件ID

        Returns:
            bool: 是否成功取消注册
        """
        with self.lock:
            if component_id not in self.collectors:
                logger.warning(
                    f"尝试取消注册不存在的组件收集器: {component_id}")
                return False

            # 获取组件类型
            component_type = self.collectors[component_id].component_type

            # 从类型分组中移除
            if component_type in self.component_types and component_id in \
                    self.component_types[component_type]:
                self.component_types[component_type].remove(component_id)
                if not self.component_types[component_type]:
                    del self.component_types[component_type]

            # 从标签索引中移除
            for tag_key, tag_values in self.tag_index.items():
                for tag_value, components in list(tag_values.items()):
                    if component_id in components:
                        components.remove(component_id)
                        if not components:
                            del tag_values[tag_value]

                if not tag_values:
                    del self.tag_index[tag_key]

            # 移除收集器
            del self.collectors[component_id]

            logger.debug(f"已取消注册组件指标收集器: {component_id}")
            return True

    def get_collector(self, component_id: str) -> Optional[
        ComponentMetricsCollector]:
        """
        获取组件指标收集器

        Args:
            component_id: 组件ID

        Returns:
            Optional[ComponentMetricsCollector]: 组件指标收集器，如果不存在则返回None
        """
        with self.lock:
            return self.collectors.get(component_id)

    def register_component(self, component_id: str,
                           component_type: str) -> ComponentMetricsCollector:
        """
        注册组件并创建指标收集器

        Args:
            component_id: 组件ID
            component_type: 组件类型

        Returns:
            ComponentMetricsCollector: 新创建的组件指标收集器
        """
        with self.lock:
            # 检查是否已存在
            if component_id in self.collectors:
                return self.collectors[component_id]

            # 创建新的收集器
            collector = ComponentMetricsCollector(component_id, component_type)

            # 注册收集器
            self.register_collector(collector)

            # 初始化常用的指标
            self._initialize_default_metrics(collector)

            return collector

    def _initialize_default_metrics(self, collector: ComponentMetricsCollector):
        """初始化组件的默认指标"""
        # 操作计数器
        collector.register_metric(MetricDefinition(
            name="operations_count",
            type=MetricType.COUNTER,
            unit=MetricUnit.OPERATIONS,
            description="组件执行的操作总数"
        ))

        # 错误计数器
        collector.register_metric(MetricDefinition(
            name="errors_count",
            type=MetricType.COUNTER,
            unit=MetricUnit.ERRORS,
            description="组件遇到的错误总数"
        ))

        # 操作耗时
        collector.register_metric(MetricDefinition(
            name="operation_time",
            type=MetricType.TIMER,
            unit=MetricUnit.MILLISECONDS,
            description="组件操作耗时",
            aggregation_methods=[AggregationMethod.AVERAGE,
                                 AggregationMethod.PERCENTILE]
        ))

        # 内存使用
        collector.register_metric(MetricDefinition(
            name="memory_usage",
            type=MetricType.GAUGE,
            unit=MetricUnit.MEGABYTES,
            description="组件内存使用量",
            threshold_warning=100,
            threshold_critical=200
        ))

        # CPU使用
        collector.register_metric(MetricDefinition(
            name="cpu_usage",
            type=MetricType.GAUGE,
            unit=MetricUnit.PERCENT,
            description="组件CPU使用率",
            threshold_warning=70,
            threshold_critical=90
        ))

    def _add_to_tag_index(self, tag_key: str, tag_value: str,
                          component_id: str):
        """添加组件到标签索引"""
        if component_id not in self.tag_index[tag_key][tag_value]:
            self.tag_index[tag_key][tag_value].append(component_id)

    def register_global_metric(self, definition: MetricDefinition) -> bool:
        """
        注册全局指标

        Args:
            definition: 指标定义

        Returns:
            bool: 是否成功注册
        """
        with self.lock:
            if definition.name in self.global_metrics:
                logger.warning(f"全局指标已存在，将被覆盖: {definition.name}")

            self.global_metrics[definition.name] = definition

            # 初始化存储
            if definition.name not in self.global_metric_values:
                self.global_metric_values[definition.name] = []

            logger.debug(f"已注册全局指标: {definition.name}")
            return True

    def record_global_metric(self, metric_name: str, value: float,
                             tags: Optional[Dict[str, str]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录全局指标值

        Args:
            metric_name: 指标名称
            value: 指标值
            tags: 标签
            metadata: 元数据

        Returns:
            bool: 是否成功记录
        """
        with self.lock:
            if metric_name not in self.global_metrics:
                logger.warning(f"尝试记录未注册的全局指标: {metric_name}")
                return False

            # 创建指标值记录
            metric_value = MetricValue(
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metadata=metadata or {}
            )

            # 添加到指标数据
            self.global_metric_values[metric_name].append(metric_value)

            return True

    def aggregate_component_metrics(self, metric_name: str,
                                    component_filter: Optional[
                                        Dict[str, str]] = None,
                                    method: AggregationMethod = AggregationMethod.AVERAGE,
                                    time_window: Optional[float] = None) -> \
    Dict[str, float]:
        """
        聚合组件指标

        Args:
            metric_name: 指标名称
            component_filter: 组件过滤条件 {tag_key: tag_value}
            method: 聚合方法
            time_window: 时间窗口（秒）

        Returns:
            Dict[str, float]: 组件ID到聚合值的映射
        """
        with self.lock:
            # 生成缓存键
            cache_key = f"agg_{metric_name}_{method.value}_{time_window}"
            if component_filter:
                cache_key += "_" + "_".join(
                    f"{k}={v}" for k, v in sorted(component_filter.items()))

            # 检查缓存
            if cache_key in self.aggregation_cache:
                cache_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cache_time < self.cache_expiry:
                    return self.aggregation_cache[cache_key]

            # 根据过滤条件获取组件
            component_ids = self._filter_components(component_filter)

            result = {}

            # 聚合每个组件的指标
            for component_id in component_ids:
                collector = self.collectors.get(component_id)
                if not collector:
                    continue

                # 检查指标是否存在
                if metric_name not in collector.metric_definitions:
                    continue

                # 获取聚合值
                value = collector.get_aggregated_value(metric_name, method,
                                                       time_window)
                if value is not None:
                    result[component_id] = value

            # 更新缓存
            self.aggregation_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()

            return result

    def _filter_components(self,
                           filter_tags: Optional[Dict[str, str]] = None) -> \
    List[str]:
        """
        根据过滤条件获取组件ID列表

        Args:
            filter_tags: 过滤条件 {tag_key: tag_value}

        Returns:
            List[str]: 组件ID列表
        """
        if not filter_tags:
            # 无过滤条件，返回所有组件
            return list(self.collectors.keys())

        # 使用标签索引快速过滤
        filtered_components = None

        for tag_key, tag_value in filter_tags.items():
            if tag_key in self.tag_index and tag_value in self.tag_index[
                tag_key]:
                # 获取匹配此标签的组件
                components = set(self.tag_index[tag_key][tag_value])

                if filtered_components is None:
                    filtered_components = components
                else:
                    # 取交集
                    filtered_components &= components
            else:
                # 标签不存在，无匹配组件
                return []

        return list(filtered_components) if filtered_components else []

    def calculate_system_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        计算系统级指标

        Returns:
            Dict[str, Dict[str, Any]]: 系统指标字典
        """
        with self.lock:
            system_metrics = {}

            # 按组件类型聚合
            for component_type, component_ids in self.component_types.items():
                type_metrics = {}

                # 每个组件类型的组件数量
                type_metrics['count'] = len(component_ids)

                # 聚合常用指标
                common_metrics = [
                    "operations_count", "errors_count", "operation_time",
                    "memory_usage", "cpu_usage"
                ]

                for metric_name in common_metrics:
                    # 收集每个组件的指标值
                    values = []

                    for component_id in component_ids:
                        collector = self.collectors.get(component_id)
                        if not collector:
                            continue

                        if metric_name not in collector.metric_definitions:
                            continue

                        # 获取最近值
                        latest = collector.get_latest_value(metric_name)
                        if latest:
                            values.append(latest.value)

                    if values:
                        # 计算统计信息
                        type_metrics[f"{metric_name}_total"] = sum(values)
                        type_metrics[f"{metric_name}_avg"] = statistics.mean(
                            values)
                        if len(values) > 1:
                            type_metrics[
                                f"{metric_name}_stddev"] = statistics.stdev(
                                values)
                        type_metrics[f"{metric_name}_min"] = min(values)
                        type_metrics[f"{metric_name}_max"] = max(values)

                system_metrics[component_type] = type_metrics

            # 添加整体系统信息
            system_metrics['system'] = {
                'total_components': len(self.collectors),
                'component_types': len(self.component_types),
                'timestamp': time.time()
            }

            return system_metrics

    def enable_reporting(self, enabled: bool = True, interval: int = 3600,
                         output_dir: str = 'metrics_reports'):
        """
        启用或禁用指标报告生成

        Args:
            enabled: 是否启用
            interval: 报告间隔（秒）
            output_dir: 输出目录
        """
        with self.lock:
            # 更新配置
            self.report_config['enabled'] = enabled
            self.report_config['interval'] = interval
            self.report_config['output_dir'] = output_dir

            # 确保输出目录存在
            if enabled and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 启动或停止报告线程
            if enabled and not self.running:
                self.running = True
                self.report_thread = threading.Thread(
                    target=self._report_loop,
                    name="MetricsReporter",
                    daemon=True
                )
                self.report_thread.start()
                logger.info(f"已启动指标报告生成，间隔: {interval}秒")
            elif not enabled and self.running:
                self.running = False
                if self.report_thread:
                    self.report_thread.join(timeout=1.0)
                    self.report_thread = None
                logger.info("已停止指标报告生成")

    def _report_loop(self):
        """报告生成循环"""
        while self.running:
            try:
                # 生成报告
                self.generate_report()

                # 等待下一次报告
                for _ in range(int(self.report_config['interval'] / 10)):
                    if not self.running:
                        break
                    time.sleep(10)

            except Exception as e:
                logger.error(f"生成指标报告出错: {e}")
                time.sleep(60)  # 出错后等待较长时间

    def generate_report(self) -> Optional[str]:
        """
        生成指标报告

        Returns:
            Optional[str]: 报告文件路径，失败返回None
        """
        with self.lock:
            try:
                # 生成报告文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.report_config['output_dir']
                filename = f"metrics_report_{timestamp}.json"
                filepath = os.path.join(output_dir, filename)

                # 准备报告数据
                report_data = {
                    'timestamp': time.time(),
                    'readable_time': datetime.now().isoformat(),
                    'system_metrics': self.calculate_system_metrics(),
                    'components': {}
                }

                # 添加每个组件的摘要
                for component_id, collector in self.collectors.items():
                    report_data['components'][
                        component_id] = collector.get_metrics_summary()

                # 添加异常信息
                if self.report_config['include_anomalies']:
                    anomalies = {}
                    for component_id, collector in self.collectors.items():
                        component_anomalies = collector.get_anomalies()
                        if any(anomalies for anomalies in
                               component_anomalies.values()):
                            anomalies[component_id] = component_anomalies

                    if anomalies:
                        report_data['anomalies'] = anomalies

                # 写入报告
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2)

                logger.info(f"已生成指标报告: {filepath}")
                return filepath

            except Exception as e:
                logger.error(f"生成指标报告失败: {e}")
                return None

    def get_anomaly_report(self) -> Dict[str, Any]:
        """
        获取系统异常报告

        Returns:
            Dict[str, Any]: 异常报告
        """
        with self.lock:
            report = {
                'timestamp': time.time(),
                'components': {}
            }

            # 收集每个组件的异常
            anomaly_count = 0

            for component_id, collector in self.collectors.items():
                # 获取组件异常
                component_anomalies = collector.get_anomalies()
                if any(anomalies for anomalies in component_anomalies.values()):
                    # 有异常的组件
                    report['components'][component_id] = {
                        'component_type': collector.component_type,
                        'anomalies': component_anomalies
                    }

                    # 统计异常总数
                    for metric_anomalies in component_anomalies.values():
                        anomaly_count += len(metric_anomalies)

            report['anomaly_count'] = anomaly_count

            return report

    def clear_caches(self):
        """清除缓存"""
        with self.lock:
            self.aggregation_cache.clear()
            self.cache_timestamps.clear()
            logger.debug("已清除指标聚合缓存")

    def shutdown(self):
        """关闭指标管理器"""
        with self.lock:
            # 停止报告生成
            self.enable_reporting(False)

            # 导出所有组件指标数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = 'metrics_export'
            os.makedirs(output_dir, exist_ok=True)

            for component_id, collector in self.collectors.items():
                filepath = os.path.join(output_dir,
                                        f"metrics_{component_id}_{timestamp}.json")
                collector.export_metrics(filepath)

            # 清除管理器状态
            self.collectors.clear()
            self.global_metrics.clear()
            self.global_metric_values.clear()
            self.component_types.clear()
            self.tag_index.clear()
            self.aggregation_cache.clear()

            logger.info("组件指标管理器已关闭")


# 辅助函数和修饰器

def with_metrics(metric_name: str,
                 component: Optional[ComponentInterface] = None):
    """
    指标记录装饰器

    Args:
        metric_name: 指标名称
        component: 组件实例，如果为None则使用第一个参数作为self

    示例:
        @with_metrics("process_data")
        def process_data(self, data):
            # 处理数据
            return result

        @with_metrics("validate_config", my_component)
        def validate_config(config):
            # 验证配置
            return is_valid
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取组件实例
            comp = component
            if comp is None and args:
                # 假设第一个参数是self
                comp = args[0]

            if not comp or not hasattr(comp, 'get_component_id'):
                # 如果没有组件，直接执行原函数
                return func(*args, **kwargs)

            # 获取组件ID
            component_id = comp.get_component_id()

            # 获取指标管理器和收集器
            metrics_manager = ComponentMetricsManager.get_instance()
            collector = metrics_manager.get_collector(component_id)

            if not collector:
                # 如果没有收集器，直接执行原函数
                return func(*args, **kwargs)

            # 获取指标名称
            full_metric_name = f"{metric_name}_time"

            # 注册指标（如果未注册）
            if full_metric_name not in collector.metric_definitions:
                collector.register_metric(MetricDefinition(
                    name=full_metric_name,
                    type=MetricType.TIMER,
                    unit=MetricUnit.MILLISECONDS,
                    description=f"{metric_name} 操作耗时"
                ))

                # 注册操作计数器
                op_metric_name = f"{metric_name}_count"
                collector.register_metric(MetricDefinition(
                    name=op_metric_name,
                    type=MetricType.COUNTER,
                    unit=MetricUnit.OPERATIONS,
                    description=f"{metric_name} 操作次数"
                ))

            # 开始计时
            timer_id = collector.start_timer(full_metric_name)

            try:
                # 执行原函数
                result = func(*args, **kwargs)

                # 增加操作计数
                collector.increment(f"{metric_name}_count")

                return result
            except Exception as e:
                # 记录错误
                collector.increment("errors_count")

                # 添加错误标签
                error_tags = {'error_type': type(e).__name__}

                # 重新抛出异常
                raise
            finally:
                # 停止计时
                if timer_id:
                    collector.stop_timer(timer_id)

        return wrapper

    return decorator


def create_metrics_collector(
        component: ComponentInterface) -> ComponentMetricsCollector:
    """
    为组件创建指标收集器

    Args:
        component: 组件实例

    Returns:
        ComponentMetricsCollector: 组件指标收集器
    """
    # 获取组件信息
    component_id = component.get_component_id()
    component_type = component.get_component_type()

    # 获取指标管理器
    metrics_manager = ComponentMetricsManager.get_instance()

    # 注册组件并获取收集器
    collector = metrics_manager.register_component(component_id, component_type)

    return collector


# 便捷函数 - 获取组件指标管理器单例
def get_metrics_manager() -> ComponentMetricsManager:
    """获取组件指标管理器单例实例"""
    return ComponentMetricsManager.get_instance()
