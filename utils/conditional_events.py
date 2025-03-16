# -*- coding: utf-8 -*-
"""
条件性事件订阅功能 - 事件系统的扩展模块 (优化版)

本模块扩展了现有事件系统，提供:
1. 基于条件表达式的事件订阅
2. 属性匹配过滤器
3. 复合条件构建器
4. 链式API支持
5. UI事件专用条件和优化
6. 高性能条件评估缓存
7. 条件对象池复用
8. 条件评估追踪和统计
9. 条件可视化支持
"""
import time
import logging
import functools
import threading
import hashlib
import json
import weakref
import gc
import sys
import random
from functools import partial
from collections import deque, Counter

# 导入事件系统常量和类
try:
    from utils.event_system import (
        EnhancedEventSystem,
        EVENT_CATEGORY_UI,
        UI_EVENT_CLICK,
        UI_EVENT_HOVER,
        UI_EVENT_DRAG,
        UI_EVENT_KEY_PRESS,
        UI_EVENT_DISPLAY_UPDATE,
        UI_EVENT_FEATURE_TOGGLE
    )
except ImportError:
    try:
        from event_system import EnhancedEventSystem

        # 如果导入成功但没有UI事件类别常量，定义它们
        EVENT_CATEGORY_UI = "ui"
        UI_EVENT_CLICK = "click"
        UI_EVENT_HOVER = "hover"
        UI_EVENT_DRAG = "drag"
        UI_EVENT_KEY_PRESS = "key_pressed"
        UI_EVENT_DISPLAY_UPDATE = "display_updated"
        UI_EVENT_FEATURE_TOGGLE = "feature_toggled"
    except ImportError:
        # 在无法导入时提供一个基本接口，以便代码能够编译
        class EnhancedEventSystem:
            def __init__(self): pass

            def subscribe(self, event_type, handler): pass

            def subscribe_with_filter(self, event_type, handler,
                                      filter_func): pass

            def subscribe_with_priority(self, event_type, handler, priority,
                                        filter_func): pass


        # 定义基本常量
        EVENT_CATEGORY_UI = "ui"
        UI_EVENT_CLICK = "click"
        UI_EVENT_HOVER = "hover"
        UI_EVENT_DRAG = "drag"
        UI_EVENT_KEY_PRESS = "key_pressed"
        UI_EVENT_DISPLAY_UPDATE = "display_updated"
        UI_EVENT_FEATURE_TOGGLE = "feature_toggled"

# 获取日志记录器
try:
    from utils.logger_config import setup_logger

    logger = setup_logger("ConditionalEventSystem")
except ImportError:
    # 基本日志记录器
    logger = logging.getLogger("ConditionalEventSystem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# 条件缓存配置
CONDITION_CACHE_SIZE = 500  # 增加缓存大小，提高命中率
CONDITION_CACHE_TIMEOUT = 10.0  # 增加超时时间，减少清理频率
CONDITION_CACHE_CLEANUP_INTERVAL = 30.0  # 设置清理间隔

# 对象池配置
CONDITION_POOL_SIZE = 100  # 条件对象池大小
CONDITION_POOL_ENABLED = True  # 是否启用对象池

# 调试配置
ENABLE_CONDITION_TRACING = False  # 是否启用条件追踪
MAX_TRACE_RECORDS = 1000  # 最大追踪记录数
PROFILING_SAMPLE_RATE = 0.01  # 性能分析采样率 (1% 的评估会被分析)

# 全局条件缓存
_condition_cache = {}
_condition_cache_lock = threading.RLock()
_cache_last_cleanup = time.time()
_cache_hits = 0
_cache_misses = 0
_condition_evaluation_times = {}  # 条件评估时间统计

# 全局条件对象池
_condition_pool = []
_condition_pool_lock = threading.RLock()
_pool_hits = 0
_pool_misses = 0

# 全局条件追踪记录
_condition_traces = deque(maxlen=MAX_TRACE_RECORDS)
_condition_trace_lock = threading.RLock()
# ------ 第1部分结束 ------

# ------ 第2部分开始：全局辅助函数 ------
def _cleanup_condition_cache():
    """清理过期的条件缓存条目"""
    global _cache_last_cleanup

    current_time = time.time()
    # 如果最后一次清理在清理间隔内，不再清理
    if current_time - _cache_last_cleanup < CONDITION_CACHE_CLEANUP_INTERVAL:
        return

    with _condition_cache_lock:
        # 记录清理开始时间
        _cache_last_cleanup = current_time

        # 识别过期条目
        expired_keys = []
        total_size = len(_condition_cache)

        # 如果缓存大小远小于上限，跳过清理
        if total_size < CONDITION_CACHE_SIZE * 0.8:
            return

        for key, (result, timestamp) in _condition_cache.items():
            if current_time - timestamp > CONDITION_CACHE_TIMEOUT:
                expired_keys.append(key)

        # 删除过期条目
        for key in expired_keys:
            del _condition_cache[key]

        # 如果还是太大，清除访问时间最早的条目
        if len(_condition_cache) > CONDITION_CACHE_SIZE:
            # 按时间戳排序
            sorted_items = sorted(_condition_cache.items(),
                                  key=lambda x: x[1][1])

            # 删除最老的 20% 缓存条目
            items_to_remove = int(CONDITION_CACHE_SIZE * 0.2)
            for i in range(items_to_remove):
                if i < len(sorted_items):
                    del _condition_cache[sorted_items[i][0]]

        # 记录清理统计
        cleaned_count = len(expired_keys)
        new_size = len(_condition_cache)
        if cleaned_count > 0:
            logger.debug(f"清理条件缓存: 删除 {cleaned_count} 个过期条目, "
                         f"当前大小: {new_size}")


def get_condition_from_pool():
    """从对象池获取一个条件对象"""
    global _pool_hits, _pool_misses

    if not CONDITION_POOL_ENABLED:
        _pool_misses += 1
        return Condition()

    with _condition_pool_lock:
        if not _condition_pool:
            _pool_misses += 1
            return Condition()

        condition = _condition_pool.pop()
        _pool_hits += 1

        # 重置条件
        condition.conditions = []
        condition.current_join = None
        condition.cache_key = None
        condition.cache_enabled = True
        condition.metadata = {}

        return condition


def return_condition_to_pool(condition):
    """将条件对象归还到对象池"""
    if not CONDITION_POOL_ENABLED:
        return False

    # 确保是有效的条件对象
    if not isinstance(condition, Condition):
        return False

    with _condition_pool_lock:
        # 避免对象池过大
        if len(_condition_pool) < CONDITION_POOL_SIZE:
            _condition_pool.append(condition)
            return True

    return False


def add_condition_trace(condition_id, data, result, evaluation_time):
    """添加条件评估记录到追踪历史"""
    if not ENABLE_CONDITION_TRACING:
        return

    with _condition_trace_lock:
        trace_entry = {
            'condition_id': condition_id,
            'data_preview': str(data)[:100] if data else None,
            'result': result,
            'evaluation_time': evaluation_time,
            'timestamp': time.time()
        }
        _condition_traces.append(trace_entry)


def get_condition_traces(condition_id=None, limit=None):
    """获取条件评估追踪历史"""
    with _condition_trace_lock:
        if condition_id:
            traces = [t for t in _condition_traces
                      if t['condition_id'] == condition_id]
        else:
            traces = list(_condition_traces)

        if limit and limit > 0:
            traces = traces[-limit:]

        return traces


def get_cache_stats():
    """获取缓存统计信息"""
    with _condition_cache_lock:
        stats = {
            'size': len(_condition_cache),
            'max_size': CONDITION_CACHE_SIZE,
            'hits': _cache_hits,
            'misses': _cache_misses,
            'hit_rate': _cache_hits / max(1, _cache_hits + _cache_misses),
            'last_cleanup': _cache_last_cleanup
        }

    return stats


def get_pool_stats():
    """获取对象池统计信息"""
    with _condition_pool_lock:
        stats = {
            'size': len(_condition_pool),
            'max_size': CONDITION_POOL_SIZE,
            'enabled': CONDITION_POOL_ENABLED,
            'hits': _pool_hits,
            'misses': _pool_misses,
            'hit_rate': _pool_hits / max(1, _pool_hits + _pool_misses)
        }

    return stats


def get_evaluation_stats():
    """获取条件评估性能统计"""
    stats = {
        'total_conditions': len(_condition_evaluation_times),
        'total_evaluations': sum(
            count for _, count in _condition_evaluation_times.values()),
        'avg_evaluation_time': 0,
        'max_evaluation_time': 0,
        'conditions_by_time': []
    }

    if _condition_evaluation_times:
        # 计算平均评估时间
        total_time = sum(total_time for total_time, _ in
                         _condition_evaluation_times.values())
        total_count = sum(
            count for _, count in _condition_evaluation_times.values())
        if total_count > 0:
            stats['avg_evaluation_time'] = total_time / total_count

        # 找出最大评估时间
        stats['max_evaluation_time'] = max(
            (total_time / count if count > 0 else 0)
            for total_time, count in _condition_evaluation_times.values()
        )

        # 获取最耗时的条件
        condition_times = []
        for condition_id, (
        total_time, count) in _condition_evaluation_times.items():
            if count > 0:
                avg_time = total_time / count
                condition_times.append((condition_id, avg_time, count))

        # 按平均时间降序排序
        condition_times.sort(key=lambda x: x[1], reverse=True)

        # 只保留前 10 个最耗时的条件
        stats['conditions_by_time'] = condition_times[:10]

    return stats


def collect_garbage():
    """手动执行垃圾回收，针对大型应用场景"""
    start_time = time.time()
    collected = gc.collect()
    elapsed = time.time() - start_time

    logger.debug(f"垃圾回收完成: 回收 {collected} 个对象, 耗时 {elapsed:.3f}秒")
    return collected
# ------ 第2部分结束 ------

# ------ 第3部分开始：Condition 类 (第一部分) ------
class Condition:
    """
    条件表达式构建器，用于创建事件过滤条件

    提供直观的API来构建复合条件表达式，支持链式调用和条件缓存。
    """

    def __init__(self, field=None, operator=None, value=None):
        """
        初始化条件构建器

        Args:
            field: 要检查的字段名
            operator: 比较操作符
            value: 比较值
        """
        self.conditions = []
        self.current_join = None
        self.cache_key = None
        self.cache_enabled = True  # 是否启用缓存
        self.metadata = {}  # 存储额外元数据
        self.last_evaluated = 0  # 最后评估时间
        self.evaluation_count = 0  # 评估次数
        self.avg_evaluation_time = 0  # 平均评估时间

        # 创建一个唯一ID用于缓存键生成和跟踪
        self.condition_id = hashlib.md5(
            f"{id(self)}:{time.time()}:{field}:{operator}".encode()
        ).hexdigest()[:8]

        if field is not None:
            self.conditions.append({
                'field': field,
                'operator': operator,
                'value': value,
                'join': None,
                'cost': self._estimate_condition_cost(field, operator,
                                                      value)
            })

    def evaluate(self, data):
        """
        评估条件是否匹配数据

        Args:
            data: 要检查的数据对象

        Returns:
            bool: 条件是否匹配
        """
        if not self.conditions:
            return True

        # 记录评估开始时间（用于性能分析）
        start_time = time.time() if ENABLE_CONDITION_TRACING or PROFILING_SAMPLE_RATE > 0 else 0
        should_profile = start_time > 0 and (ENABLE_CONDITION_TRACING or
                                             random.random() < PROFILING_SAMPLE_RATE)

        # 尝试使用缓存（如果启用）
        if self.cache_enabled:
            cache_key = self._generate_cache_key(data)
            if cache_key:
                with _condition_cache_lock:
                    if cache_key in _condition_cache:
                        result, timestamp = _condition_cache[cache_key]
                        # 更新时间戳以保持活跃
                        _condition_cache[cache_key] = (result, time.time())
                        global _cache_hits
                        _cache_hits += 1

                        # 记录性能统计
                        if should_profile:
                            evaluation_time = time.time() - start_time
                            add_condition_trace(self.condition_id, data,
                                                result,
                                                evaluation_time)

                            # 更新性能统计数据
                            if self.condition_id in _condition_evaluation_times:
                                total_time, count = \
                                _condition_evaluation_times[
                                    self.condition_id]
                                _condition_evaluation_times[
                                    self.condition_id] = (
                                    total_time + evaluation_time, count + 1
                                )
                            else:
                                _condition_evaluation_times[
                                    self.condition_id] = (
                                    evaluation_time, 1
                                )

                        return result

                global _cache_misses
                _cache_misses += 1

        # 按照成本排序条件 - 低成本条件优先评估
        if len(self.conditions) > 1:
            sorted_conditions = sorted(self.conditions,
                                       key=lambda c: c.get('cost',
                                                           float('inf')))
        else:
            sorted_conditions = self.conditions

        # 计算结果 - 使用短路逻辑
        first_condition = sorted_conditions[0]
        result = self._evaluate_condition(first_condition, data)

        # 如果第一个条件是AND链接，且结果为False，可以直接返回False（短路）
        if not result and len(sorted_conditions) > 1 and sorted_conditions[
            1].get('join') == 'AND':
            # 更新统计数据
            if should_profile:
                evaluation_time = time.time() - start_time
                add_condition_trace(self.condition_id, data, False,
                                    evaluation_time)

                # 更新性能统计数据
                if self.condition_id in _condition_evaluation_times:
                    total_time, count = _condition_evaluation_times[
                        self.condition_id]
                    _condition_evaluation_times[self.condition_id] = (
                        total_time + evaluation_time, count + 1
                    )
                else:
                    _condition_evaluation_times[self.condition_id] = (
                    evaluation_time, 1)

            # 缓存结果
            if self.cache_enabled and cache_key:
                with _condition_cache_lock:
                    _condition_cache[cache_key] = (False, time.time())

            return False

        # 继续评估剩余条件
        for i in range(1, len(sorted_conditions)):
            condition = sorted_conditions[i]
            join = condition['join']

            # 应用短路逻辑
            if (join == 'AND' and not result) or (join == 'OR' and result):
                break

            cond_result = self._evaluate_condition(condition, data)

            if join == 'AND':
                result = result and cond_result
            elif join == 'OR':
                result = result or cond_result
            else:
                # 默认为AND
                result = result and cond_result

        # 缓存结果（如果启用）
        if self.cache_enabled and cache_key:
            with _condition_cache_lock:
                # 定期清理缓存
                if random.random() < 0.01:  # 只有1%的机会触发清理
                    _cleanup_condition_cache()
                # 存储结果和时间戳
                _condition_cache[cache_key] = (result, time.time())

        # 更新统计数据
        if should_profile:
            evaluation_time = time.time() - start_time
            add_condition_trace(self.condition_id, data, result,
                                evaluation_time)

            # 更新自身的性能统计
            self.evaluation_count += 1
            self.last_evaluated = time.time()

            # 更新平均评估时间
            self.avg_evaluation_time = (
                    (self.avg_evaluation_time * (
                                self.evaluation_count - 1) + evaluation_time) /
                    self.evaluation_count
            )

            # 更新全局性能统计数据
            if self.condition_id in _condition_evaluation_times:
                total_time, count = _condition_evaluation_times[
                    self.condition_id]
                _condition_evaluation_times[self.condition_id] = (
                    total_time + evaluation_time, count + 1
                )
            else:
                _condition_evaluation_times[self.condition_id] = (
                evaluation_time, 1)

        return result
# ------ 第3部分结束 ------

    # ------ 第4部分开始：Condition 类 (第二部分) ------
    def _estimate_condition_cost(self, field, operator, value):
        """
        估算条件评估成本（用于优化评估顺序）

        Args:
            field: 字段名
            operator: 操作符
            value: 比较值

        Returns:
            float: 估算成本（值越小表示成本越低）
        """
        # 基本成本
        cost = 1.0

        # 嵌套字段路径成本更高
        if field and '.' in field:
            cost += 0.5 * field.count('.')

        # 操作符成本
        if operator in ['==', '!=', 'in', 'not_in']:
            # 简单比较成本低
            cost += 0.1
        elif operator in ['contains', 'starts_with', 'ends_with']:
            # 字符串操作成本中等
            cost += 0.3
        elif operator in ['in_area', 'near']:
            # 空间计算成本高
            cost += 0.5
        elif operator in ['matches']:
            # 回调函数成本很高
            cost += 0.8
        elif operator in ['any_of', 'all_of']:
            # 复合条件成本最高
            cost += 1.0

        # 值类型的成本
        if value is None:
            # None比较简单
            cost += 0.0
        elif isinstance(value, (int, float, bool, str)):
            # 基本类型比较简单
            cost += 0.1
        elif callable(value):
            # 可调用对象成本高
            cost += 0.8
        elif isinstance(value, (list, tuple)):
            # 集合类型成本取决于大小
            cost += 0.2 * min(1.0, len(value) / 20)

        return cost

    def _generate_cache_key(self, data):
        """
        为条件评估结果生成缓存键

        Args:
            data: 事件数据

        Returns:
            str: 缓存键，或None表示不应缓存
        """
        # 只为特定事件启用缓存
        event_type = data.get('event_type', '')

        # 优化：只缓存频繁的UI事件
        cacheable_events = {
            UI_EVENT_CLICK, UI_EVENT_HOVER, UI_EVENT_KEY_PRESS,
            UI_EVENT_DISPLAY_UPDATE, UI_EVENT_FEATURE_TOGGLE, UI_EVENT_DRAG
        }

        if not any(event_type.startswith(e) for e in cacheable_events):
            return None

        # 从关键数据字段创建哈希
        try:
            # 如果已经有缓存键且结构未变，直接使用
            if self.cache_key and not hasattr(self, '_data_schema_changed'):
                base_key = self.cache_key
            else:
                # 生成表示条件结构的基础键
                condition_hash = hashlib.md5(
                    json.dumps([(c.get('field'), c.get('operator'),
                                 str(type(c.get('value'))))
                                for c in self.conditions],
                               sort_keys=True).encode()
                ).hexdigest()[:16]

                base_key = f"cond:{self.condition_id}:{condition_hash}"
                self.cache_key = base_key

            # 从数据中提取关键字段作为键的一部分
            data_key_parts = [f"event:{event_type}"]

            # 根据事件类型选择关键字段
            if event_type.startswith(
                    UI_EVENT_CLICK) or event_type.startswith(
                    UI_EVENT_HOVER):
                # 对于点击和悬停事件，缓存键包含元素ID和大致位置
                if 'element' in data:
                    data_key_parts.append(f"elem:{data['element']}")
                if 'x' in data and 'y' in data:
                    # 量化坐标，避免过多的缓存键
                    grid_size = 20  # 增大网格大小，减少唯一键数量
                    x_grid = data['x'] // grid_size
                    y_grid = data['y'] // grid_size
                    data_key_parts.append(f"pos:{x_grid}_{y_grid}")
            elif event_type.startswith(UI_EVENT_KEY_PRESS):
                # 对于按键事件，缓存键包含按键
                if 'key' in data:
                    data_key_parts.append(f"key:{data['key']}")
            elif event_type.startswith(UI_EVENT_FEATURE_TOGGLE):
                # 对于功能切换事件，缓存键包含功能名称和状态
                if 'feature_name' in data:
                    data_key_parts.append(f"feature:{data['feature_name']}")
                if 'state' in data:
                    data_key_parts.append(f"state:{data['state']}")

            # 计算数据部分的哈希
            data_hash = hashlib.md5(
                ":".join(data_key_parts).encode()).hexdigest()[:8]

            # 完整缓存键
            return f"{base_key}:{data_hash}"
        except:
            return None

    def _evaluate_condition(self, condition, data):
        """
        评估单个条件

        Args:
            condition: 条件字典
            data: 数据对象

        Returns:
            bool: 条件是否匹配
        """
        if not isinstance(data, dict):
            return False

        field = condition['field']
        operator = condition['operator']
        expected_value = condition['value']

        # 对嵌套字段路径进行优化处理
        if '.' in field:
            actual_value = self._get_nested_field(data, field)
            if actual_value is None:
                return False
        else:
            # 简单字段
            if field not in data:
                return False
            actual_value = data[field]

        # 执行比较
        return self._compare_values(operator, actual_value, expected_value,
                                    data)

    def _get_nested_field(self, data, field_path):
        """
        获取嵌套字段值 (优化版)

        Args:
            data: 数据对象
            field_path: 字段路径，如 "user.name"

        Returns:
            字段值或None（如果不存在）
        """
        if '.' not in field_path:
            return data.get(field_path)

        parts = field_path.split('.')
        curr = data

        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None

        return curr
    # ------ 第4部分结束 ------

    # ------ 第5部分开始：Condition 类 (第三部分) ------
    def _compare_values(self, operator, actual_value, expected_value, data):
        """
        比较值 (优化版)

        Args:
            operator: 比较操作符
            actual_value: 实际值
            expected_value: 期望值
            data: 完整数据对象 (用于某些操作符)

        Returns:
            bool: 比较结果
        """
        # 优化常见操作符路径
        if operator == '==':
            return actual_value == expected_value
        elif operator == '!=':
            return actual_value != expected_value
        elif operator == 'in':
            return actual_value in expected_value
        elif operator == 'not_in':
            return actual_value not in expected_value

        # 需要类型检查的操作符
        if operator in ['>', '>=', '<', '<=']:
            # 尝试数值比较
            try:
                if operator == '>':
                    return float(actual_value) > float(expected_value)
                elif operator == '>=':
                    return float(actual_value) >= float(expected_value)
                elif operator == '<':
                    return float(actual_value) < float(expected_value)
                elif operator == '<=':
                    return float(actual_value) <= float(expected_value)
            except (TypeError, ValueError):
                # 如果无法转换为数值，则使用字符串比较
                try:
                    actual_str = str(actual_value)
                    expected_str = str(expected_value)
                    if operator == '>':
                        return actual_str > expected_str
                    elif operator == '>=':
                        return actual_str >= expected_str
                    elif operator == '<':
                        return actual_str < expected_str
                    elif operator == '<=':
                        return actual_str <= expected_str
                except:
                    return False

        # 字符串操作符
        elif operator in ['contains', 'starts_with', 'ends_with']:
            try:
                if operator == 'contains':
                    return expected_value in actual_value
                elif operator == 'starts_with':
                    return str(actual_value).startswith(str(expected_value))
                elif operator == 'ends_with':
                    return str(actual_value).endswith(str(expected_value))
            except:
                return False

        # 回调操作符
        elif operator == 'matches':
            if callable(expected_value):
                try:
                    return expected_value(actual_value)
                except:
                    return False
            return False

        # UI特定的操作符 - 优化坐标操作符
        elif operator in ['in_area', 'near']:
            return self._evaluate_spatial_operator(operator, actual_value,
                                                   expected_value, data)

        # 复合操作符
        elif operator in ['any_of', 'all_of']:
            return self._evaluate_compound_operator(operator, actual_value,
                                                    expected_value, data)

        # 未知操作符
        else:
            logger.warning(f"未知操作符: {operator}")
            return False

    def _evaluate_spatial_operator(self, operator, actual_value,
                                   expected_value,
                                   data):
        """
        评估空间操作符（in_area, near）

        Args:
            operator: 操作符
            actual_value: 实际值
            expected_value: 期望值
            data: 完整数据对象

        Returns:
            bool: 评估结果
        """
        x, y = None, None

        # 尝试从不同来源获取坐标
        if isinstance(actual_value, (list, tuple)) and len(
                actual_value) >= 2:
            x, y = actual_value[0], actual_value[1]
        elif 'x' in data and 'y' in data:
            x, y = data['x'], data['y']
        else:
            return False

        if operator == 'in_area':
            # 检查坐标是否在指定区域内
            try:
                x1, y1, x2, y2 = expected_value
                return x1 <= x <= x2 and y1 <= y <= y2
            except:
                return False
        elif operator == 'near':
            # 检查坐标是否接近指定点
            try:
                target_x, target_y, distance = expected_value
                # 使用平方距离比较，避免开平方操作
                return ((x - target_x) ** 2 + (
                        y - target_y) ** 2) <= distance ** 2
            except:
                return False

        return False

    def _evaluate_compound_operator(self, operator, actual_value,
                                    expected_value, data):
        """
        评估复合操作符（any_of, all_of）

        Args:
            operator: 操作符
            actual_value: 实际值 (这里通常不使用)
            expected_value: 子条件列表
            data: 完整数据对象

        Returns:
            bool: 评估结果
        """
        if not isinstance(expected_value, (list, tuple)):
            return False

        if operator == 'any_of':
            # 任一条件满足
            for cond in expected_value:
                if isinstance(cond, Condition) and cond.evaluate(data):
                    return True
            return False
        elif operator == 'all_of':
            # 所有条件都满足
            if not expected_value:
                return False

            for cond in expected_value:
                if not isinstance(cond, Condition) or not cond.evaluate(
                        data):
                    return False
            return True
# ------ 第5部分结束 ------

    # ------ 第6部分开始：Condition 类 (第四部分) ------
    def __call__(self, data):
        """
        使对象可调用，方便作为过滤函数

        Args:
            data: 要检查的数据

        Returns:
            bool: 条件是否匹配
        """
        return self.evaluate(data)

    def get_stats(self):
        """获取条件统计信息"""
        return {
            'id': self.condition_id,
            'conditions_count': len(self.conditions),
            'evaluation_count': self.evaluation_count,
            'avg_evaluation_time': self.avg_evaluation_time,
            'last_evaluated': self.last_evaluated,
            'cache_enabled': self.cache_enabled
        }

    def disable_cache(self):
        """禁用此条件的缓存"""
        self.cache_enabled = False
        return self

    def enable_cache(self):
        """启用此条件的缓存"""
        self.cache_enabled = True
        return self

    def set_metadata(self, key, value):
        """设置条件元数据"""
        self.metadata[key] = value
        return self

    def get_metadata(self, key, default=None):
        """获取条件元数据"""
        return self.metadata.get(key, default)

    def destroy(self):
        """销毁条件对象，回收资源"""
        # 尝试归还到对象池
        if return_condition_to_pool(self):
            return True

        # 手动清理大型字段
        self.conditions = []
        self.metadata = {}
        return False

    # 比较运算符方法
    def equals(self, field, value):
        """字段等于指定值"""
        return self._add_condition(field, '==', value)

    def not_equals(self, field, value):
        """字段不等于指定值"""
        return self._add_condition(field, '!=', value)

    def greater_than(self, field, value):
        """字段大于指定值"""
        return self._add_condition(field, '>', value)

    def greater_equals(self, field, value):
        """字段大于等于指定值"""
        return self._add_condition(field, '>=', value)

    def less_than(self, field, value):
        """字段小于指定值"""
        return self._add_condition(field, '<', value)

    def less_equals(self, field, value):
        """字段小于等于指定值"""
        return self._add_condition(field, '<=', value)

    def is_in(self, field, values):
        """字段值在集合中"""
        return self._add_condition(field, 'in', values)

    def not_in(self, field, values):
        """字段值不在集合中"""
        return self._add_condition(field, 'not_in', values)

    def contains(self, field, value):
        """字段值包含指定值"""
        return self._add_condition(field, 'contains', value)

    def starts_with(self, field, prefix):
        """字段值以指定前缀开始"""
        return self._add_condition(field, 'starts_with', prefix)

    def ends_with(self, field, suffix):
        """字段值以指定后缀结束"""
        return self._add_condition(field, 'ends_with', suffix)

    def matches(self, field, validator):
        """字段值满足指定验证函数"""
        return self._add_condition(field, 'matches', validator)

    def exists(self, field):
        """字段存在"""
        return self._add_condition(field, 'matches',
                                   lambda x: x is not None)

    # UI特定的条件方法
    def in_area(self, field, x1, y1, x2, y2):
        """
        检查坐标是否在指定区域内

        Args:
            field: 坐标字段，如'position'，或直接使用事件的x、y字段
            x1, y1: 左上角坐标
            x2, y2: 右下角坐标
        """
        return self._add_condition(field, 'in_area', (x1, y1, x2, y2))

    def near(self, field, x, y, distance):
        """
        检查坐标是否在指定点附近

        Args:
            field: 坐标字段
            x, y: 目标点坐标
            distance: 最大距离
        """
        return self._add_condition(field, 'near', (x, y, distance))

    def any_of(self, field, conditions):
        """
        检查是否满足任一条件

        Args:
            field: 字段名
            conditions: 条件列表
        """
        return self._add_condition(field, 'any_of', conditions)

    def all_of(self, field, conditions):
        """
        检查是否满足所有条件

        Args:
            field: 字段名
            conditions: 条件列表
        """
        return self._add_condition(field, 'all_of', conditions)

    def element_is(self, element_id):
        """元素ID等于指定值"""
        return self.equals('element', element_id)

    def element_type_is(self, element_type):
        """元素类型等于指定值"""
        return self.equals('element_type', element_type)

    def key_is(self, key):
        """按键等于指定值"""
        return self.equals('key', key)

    # 逻辑连接符
    def AND(self):
        """与运算连接符"""
        self.current_join = 'AND'
        return self

    def OR(self):
        """或运算连接符"""
        self.current_join = 'OR'
        return self

    def _add_condition(self, field, operator, value):
        """
        添加条件

        Args:
            field: 字段名
            operator: 操作符
            value: 比较值

        Returns:
            self: 支持链式调用
        """
        condition = {
            'field': field,
            'operator': operator,
            'value': value,
            'join': self.current_join,
            'cost': self._estimate_condition_cost(field, operator, value)
        }

        self.conditions.append(condition)
        self.current_join = None

        # 清除缓存键，因为条件已变更
        self.cache_key = None
        # 标记数据模式已改变
        self._data_schema_changed = True

        return self
# ------ 第6部分结束 ------

# ------ 第7a部分开始：ConditionalEventSystem 类（第一部分） ------
class ConditionalEventSystem:
    """
    条件性事件系统

    扩展EnhancedEventSystem，添加条件式订阅支持，
    使用更直观的API进行事件过滤和优化性能。
    """

    def __init__(self, base_event_system=None):
        """
        初始化条件性事件系统

        Args:
            base_event_system: 基础事件系统实例，如果为None则创建新实例
        """
        # 使用提供的事件系统或创建新的
        self.event_system = base_event_system or EnhancedEventSystem()

        # 跟踪订阅
        self.subscriptions = {}  # 事件类型 -> [条件列表]
        self.subscription_stats = {
            'total': 0,
            'ui_events': 0,
            'conditional': 0,
            'batched': 0
        }

        # 条件评估统计
        self.eval_stats = {
            'evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'ui_evaluations': 0,
            'total_evaluation_time': 0
        }

        # 批处理配置
        self.batch_processing = True  # 是否启用批处理
        self.batch_size = 10  # 批处理大小
        self.batch_interval = 0.05  # 批处理间隔(秒)
        self.batched_events = {}  # 事件类型 -> 事件列表
        self.batch_timer = None  # 批处理定时器
        self.batch_lock = threading.RLock()  # 批处理锁

        # 锁
        self.lock = threading.RLock()

        # 启用性能监控
        self.performance_monitoring = True

        # 启用批处理
        self._start_batch_processing()

        logger.info("条件性事件系统已初始化")

    def set_batch_processing(self, enabled, batch_size=None,
                             batch_interval=None):
        """
        设置批处理配置

        Args:
            enabled: 是否启用批处理
            batch_size: 批处理大小
            batch_interval: 批处理间隔(秒)
        """
        self.batch_processing = enabled

        if batch_size is not None:
            self.batch_size = max(1, batch_size)

        if batch_interval is not None:
            self.batch_interval = max(0.01, batch_interval)

        if enabled and not self.batch_timer:
            self._start_batch_processing()
        elif not enabled and self.batch_timer:
            self._stop_batch_processing()

        logger.info(f"批处理已{'启用' if enabled else '禁用'}, "
                    f"大小={self.batch_size}, 间隔={self.batch_interval}秒")

    def _start_batch_processing(self):
        """启动批处理"""
        if not self.batch_processing:
            return

        # 创建定时器
        self.batch_timer = threading.Timer(self.batch_interval,
                                           self._process_batched_events)
        self.batch_timer.daemon = True
        self.batch_timer.start()

        logger.debug("批处理定时器已启动")

    def _stop_batch_processing(self):
        """停止批处理"""
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None

        # 处理所有未处理的批次
        self._process_batched_events()

        logger.debug("批处理定时器已停止")

    def _process_batched_events(self):
        """处理批处理队列中的事件"""
        try:
            # 获取当前批次
            events_to_process = {}

            with self.batch_lock:
                # 交换批次队列
                events_to_process, self.batched_events = self.batched_events, {}

                # 重启定时器
                if self.batch_processing:
                    self.batch_timer = threading.Timer(self.batch_interval,
                                                       self._process_batched_events)
                    self.batch_timer.daemon = True
                    self.batch_timer.start()

            # 处理每个事件类型的批次
            total_processed = 0
            for event_type, events in events_to_process.items():
                # 去重
                unique_events = {}
                for event_data in events:
                    # 使用事件特征作为键去重 (保留最新的)
                    event_key = self._get_event_key(event_type, event_data)
                    unique_events[event_key] = event_data

                # 处理唯一事件
                for event_data in unique_events.values():
                    self._direct_publish_event(event_type, event_data)
                    total_processed += 1

            if total_processed > 0:
                logger.debug(f"批处理: 处理了 {total_processed} 个事件 "
                             f"(去重后), {len(events_to_process)} 个事件类型")

        except Exception as e:
            logger.error(f"处理批次事件出错: {e}")

            # 确保定时器能继续运行
            with self.batch_lock:
                if self.batch_processing and not self.batch_timer:
                    self.batch_timer = threading.Timer(self.batch_interval,
                                                       self._process_batched_events)
                    self.batch_timer.daemon = True
                    self.batch_timer.start()

    def _get_event_key(self, event_type, event_data):
        """
        为事件生成用于去重的键

        Args:
            event_type: 事件类型
            event_data: 事件数据

        Returns:
            str: 事件键
        """
        key_parts = [event_type]

        # 根据事件类型选择关键字段
        if event_type in [UI_EVENT_CLICK, UI_EVENT_HOVER]:
            if 'element' in event_data:
                key_parts.append(f"elem:{event_data['element']}")
            if 'x' in event_data and 'y' in event_data:
                # 网格化坐标
                grid_size = 10
                x_grid = event_data['x'] // grid_size
                y_grid = event_data['y'] // grid_size
                key_parts.append(f"pos:{x_grid}_{y_grid}")
        elif event_type == UI_EVENT_KEY_PRESS:
            if 'key' in event_data:
                key_parts.append(f"key:{event_data['key']}")
        elif event_type == UI_EVENT_FEATURE_TOGGLE:
            if 'feature_name' in event_data:
                key_parts.append(f"feature:{event_data['feature_name']}")

        # 将所有部分连接成字符串
        return ":".join(key_parts)
# ------ 第7a部分结束 ------

    # ------ 第7b部分开始：ConditionalEventSystem 类（第二部分） ------
    def subscribe(self, event_type, handler, priority=0):
        """
        基本事件订阅（无条件）

        Args:
            event_type: 事件类型
            handler: 处理函数
            priority: 优先级

        Returns:
            bool: 是否成功订阅
        """
        with self.lock:
            # 跟踪订阅
            self.subscription_stats['total'] += 1
            if event_type in (UI_EVENT_CLICK, UI_EVENT_HOVER, UI_EVENT_DRAG,
                              UI_EVENT_KEY_PRESS, UI_EVENT_DISPLAY_UPDATE,
                              UI_EVENT_FEATURE_TOGGLE):
                self.subscription_stats['ui_events'] += 1

            # 添加到跟踪列表
            if event_type not in self.subscriptions:
                self.subscriptions[event_type] = []

        return self.event_system.subscribe_with_priority(
            event_type, handler, priority, None)

    def subscribe_if(self, event_type, condition, handler, priority=0):
        """
        基于条件订阅事件

        Args:
            event_type: 事件类型
            condition: Condition对象或函数
            handler: 处理函数
            priority: 优先级

        Returns:
            bool: 是否成功订阅
        """
        # 准备过滤器
        filter_func = None

        if isinstance(condition, Condition):
            # 创建性能跟踪包装器
            @functools.wraps(condition.__call__)
            def tracking_filter(data):
                start_time = time.time() if self.performance_monitoring else 0

                # 更新评估统计
                with self.lock:
                    self.eval_stats['evaluations'] += 1
                    if event_type in (
                            UI_EVENT_CLICK, UI_EVENT_HOVER, UI_EVENT_DRAG,
                            UI_EVENT_KEY_PRESS, UI_EVENT_DISPLAY_UPDATE,
                            UI_EVENT_FEATURE_TOGGLE):
                        self.eval_stats['ui_evaluations'] += 1

                # 评估条件
                result = condition(data)

                # 更新性能统计
                if self.performance_monitoring:
                    eval_time = time.time() - start_time
                    with self.lock:
                        self.eval_stats[
                            'total_evaluation_time'] += eval_time

                return result

            filter_func = tracking_filter
        elif callable(condition):
            # 直接使用可调用对象
            filter_func = condition
        else:
            logger.error(f"无效的条件类型: {type(condition)}")
            return False

        with self.lock:
            # 跟踪条件订阅
            self.subscription_stats['total'] += 1
            self.subscription_stats['conditional'] += 1
            if event_type in (UI_EVENT_CLICK, UI_EVENT_HOVER, UI_EVENT_DRAG,
                              UI_EVENT_KEY_PRESS, UI_EVENT_DISPLAY_UPDATE,
                              UI_EVENT_FEATURE_TOGGLE):
                self.subscription_stats['ui_events'] += 1

            # 添加到跟踪列表
            if event_type not in self.subscriptions:
                self.subscriptions[event_type] = []

            # 存储条件引用（如果是Condition对象）
            if isinstance(condition, Condition):
                self.subscriptions[event_type].append(
                    weakref.ref(condition))

        return self.event_system.subscribe_with_priority(
            event_type, handler, priority, filter_func)

    def subscribe_when(self, condition_builder):
        """
        返回条件构建器，用于链式API

        Args:
            condition_builder: 用于构建条件的函数

        Returns:
            EventSubscriptionBuilder: 订阅构建器
        """
        return EventSubscriptionBuilder(self, condition_builder)

    def publish(self, event_type, data=None):
        """
        发布事件

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            bool: 是否成功发布
        """
        # 对UI事件进行批处理
        if (self.batch_processing and
                event_type in (
                UI_EVENT_CLICK, UI_EVENT_HOVER, UI_EVENT_DRAG,
                UI_EVENT_KEY_PRESS, UI_EVENT_DISPLAY_UPDATE)):
            return self._batch_publish_event(event_type, data)
        else:
            # 直接发布非UI事件
            return self._direct_publish_event(event_type, data)

    def _batch_publish_event(self, event_type, data):
        """
        将事件添加到批处理队列

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            bool: 是否成功添加到队列
        """
        try:
            # 预处理数据
            if data is None:
                data = {}
            elif not isinstance(data, dict):
                data = {'value': data}

            # 确保时间戳和事件类型
            if 'timestamp' not in data:
                data['timestamp'] = time.time()
            if 'event_type' not in data:
                data['event_type'] = event_type

            # 添加到批处理队列
            with self.batch_lock:
                if event_type not in self.batched_events:
                    self.batched_events[event_type] = []

                self.batched_events[event_type].append(data)
                batch_size = sum(
                    len(events) for events in self.batched_events.values())

                # 如果已达到批处理大小，立即处理
                if batch_size >= self.batch_size:
                    # 取消现有定时器
                    if self.batch_timer:
                        self.batch_timer.cancel()
                        self.batch_timer = None

                    # 立即处理批次
                    threading.Thread(target=self._process_batched_events,
                                     daemon=True).start()

            with self.lock:
                self.subscription_stats['batched'] += 1

            return True

        except Exception as e:
            logger.error(f"批处理发布事件错误: {e}")
            # 在出错时尝试直接发布
            return self._direct_publish_event(event_type, data)

    def _direct_publish_event(self, event_type, data):
        """
        直接发布事件（不经过批处理）

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            bool: 是否成功发布
        """
        return self.event_system.publish(event_type, data)

    def unsubscribe(self, event_type, handler):
        """
        取消订阅

        Args:
            event_type: 事件类型
            handler: 处理函数

        Returns:
            bool: 是否成功取消订阅
        """
        # 无法准确跟踪取消订阅的统计，因为我们没有保存handler到condition的映射
        return self.event_system.unsubscribe(event_type, handler)

    def get_stats(self):
        """
        获取条件系统统计信息

        Returns:
            dict: 统计信息
        """
        with self.lock:
            # 获取缓存和对象池统计
            cache_stats = get_cache_stats()
            pool_stats = get_pool_stats()
            eval_stats = get_evaluation_stats()

            stats = {
                'subscriptions': dict(self.subscription_stats),
                'evaluations': dict(self.eval_stats),
                'cache': cache_stats,
                'pool': pool_stats,
                'performance': eval_stats,
                'batch': {
                    'enabled': self.batch_processing,
                    'size': self.batch_size,
                    'interval': self.batch_interval,
                    'queue_sizes': {event_type: len(events)
                                    for event_type, events in
                                    self.batched_events.items()}
                },
                'events': {
                    'types': len(self.subscriptions),
                    'list': list(self.subscriptions.keys())
                }
            }

            # 计算平均评估时间
            if self.eval_stats['evaluations'] > 0:
                stats['evaluations']['avg_time'] = (
                        self.eval_stats['total_evaluation_time'] /
                        self.eval_stats['evaluations']
                )

            return stats

    def clear_cache(self):
        """
        清除条件评估缓存

        Returns:
            int: 清除的缓存条目数
        """
        with _condition_cache_lock:
            count = len(_condition_cache)
            _condition_cache.clear()
            return count

    def reset_stats(self):
        """
        重置统计信息

        Returns:
            dict: 重置前的统计信息
        """
        with self.lock:
            old_stats = {
                'subscriptions': dict(self.subscription_stats),
                'evaluations': dict(self.eval_stats)
            }

            # 重置统计
            self.eval_stats = {
                'evaluations': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'ui_evaluations': 0,
                'total_evaluation_time': 0
            }

            return old_stats

    def shutdown(self):
        """关闭条件系统，释放资源"""
        # 停止批处理
        self._stop_batch_processing()

        # 清理缓存
        self.clear_cache()

        # 运行垃圾回收
        collect_garbage()

        logger.info("条件性事件系统已关闭")

    # 委托其他方法到基础事件系统
    def __getattr__(self, name):
        """转发未定义的方法到基础事件系统"""
        return getattr(self.event_system, name)
# ------ 第7b部分结束 ------
# ------ 第8部分开始：辅助类和函数 ------
class EventSubscriptionBuilder:
    """
    事件订阅构建器

    支持链式API进行条件式事件订阅。
    """

    def __init__(self, event_system, condition_builder):
        """
        初始化订阅构建器

        Args:
            event_system: 条件性事件系统
            condition_builder: 条件构建函数
        """
        self.event_system = event_system

        # 创建条件
        if isinstance(condition_builder, Condition):
            self.condition = condition_builder
        elif callable(condition_builder):
            try:
                # 如果是函数，执行它创建条件
                condition = condition_builder()
                if isinstance(condition, Condition):
                    self.condition = condition
                else:
                    self.condition = Condition()
                    logger.warning(
                        "条件构建器未返回有效的Condition对象，使用默认条件")
            except Exception as e:
                logger.error(f"执行条件构建器时出错: {e}")
                self.condition = Condition()
        else:
            self.condition = Condition()
            logger.warning(f"无效的条件构建器类型: {type(condition_builder)}")

    def then(self, event_type, handler, priority=0):
        """
        完成订阅设置

        Args:
            event_type: 事件类型
            handler: 处理函数
            priority: 优先级

        Returns:
            bool: 是否成功订阅
        """
        return self.event_system.subscribe_if(
            event_type, self.condition, handler, priority)


class FieldConditionBuilder:
    """
    字段条件构建器

    提供流畅的API来构建基于特定字段的条件。
    """

    def __init__(self, field):
        """
        初始化字段条件构建器

        Args:
            field: 字段名
        """
        self.field = field
        self.condition = get_condition_from_pool()  # 使用对象池

    def equals(self, value):
        """字段等于指定值"""
        return self.condition.equals(self.field, value)

    def not_equals(self, value):
        """字段不等于指定值"""
        return self.condition.not_equals(self.field, value)

    def greater_than(self, value):
        """字段大于指定值"""
        return self.condition.greater_than(self.field, value)

    def greater_equals(self, value):
        """字段大于等于指定值"""
        return self.condition.greater_equals(self.field, value)

    def less_than(self, value):
        """字段小于指定值"""
        return self.condition.less_than(self.field, value)

    def less_equals(self, value):
        """字段小于等于指定值"""
        return self.condition.less_equals(self.field, value)

    def is_in(self, values):
        """字段值在集合中"""
        return self.condition.is_in(self.field, values)

    def not_in(self, values):
        """字段值不在集合中"""
        return self.condition.not_in(self.field, values)

    def contains(self, value):
        """字段值包含指定值"""
        return self.condition.contains(self.field, value)

    def starts_with(self, prefix):
        """字段值以指定前缀开始"""
        return self.condition.starts_with(self.field, prefix)

    def ends_with(self, suffix):
        """字段值以指定后缀结束"""
        return self.condition.ends_with(self.field, suffix)

    def matches(self, validator):
        """字段值满足指定验证函数"""
        return self.condition.matches(self.field, validator)

    def exists(self):
        """字段存在"""
        return self.condition.exists(self.field)

    # UI特定方法
    def in_area(self, x1, y1, x2, y2):
        """坐标在指定区域内"""
        return self.condition.in_area(self.field, x1, y1, x2, y2)

    def near(self, x, y, distance):
        """坐标在指定点附近"""
        return self.condition.near(self.field, x, y, distance)


# UI事件特定的条件构建器
class UIConditionBuilder:
    """
    UI事件条件构建器

    提供专用于UI事件的条件构建API
    """

    @staticmethod
    def click_in_area(x1, y1, x2, y2):
        """
        创建检查点击是否在指定区域内的条件

        Args:
            x1, y1: 左上角坐标
            x2, y2: 右下角坐标

        Returns:
            Condition: 条件对象
        """
        condition = get_condition_from_pool()
        condition.equals('event_type', UI_EVENT_CLICK)
        condition.AND()
        condition.in_area('position', x1, y1, x2, y2)
        return condition

    @staticmethod
    def click_on_element(element_id):
        """
        创建检查点击是否在指定元素上的条件

        Args:
            element_id: 元素ID

        Returns:
            Condition: 条件对象
        """
        condition = get_condition_from_pool()
        condition.equals('event_type', UI_EVENT_CLICK)
        condition.AND()
        condition.element_is(element_id)
        return condition

    @staticmethod
    def hover_on_element(element_id):
        """
        创建检查悬停是否在指定元素上的条件

        Args:
            element_id: 元素ID

        Returns:
            Condition: 条件对象
        """
        condition = get_condition_from_pool()
        condition.equals('event_type', UI_EVENT_HOVER)
        condition.AND()
        condition.element_is(element_id)
        return condition

    @staticmethod
    def key_press(key=None):
        """
        创建检查按键事件的条件

        Args:
            key: 可选的键值

        Returns:
            Condition: 条件对象
        """
        condition = get_condition_from_pool()
        condition.equals('event_type', UI_EVENT_KEY_PRESS)
        if key:
            condition.AND()
            condition.key_is(key)
        return condition

    @staticmethod
    def feature_state(feature_name, state=True):
        """
        创建检查功能状态的条件

        Args:
            feature_name: 功能名称
            state: 期望的状态

        Returns:
            Condition: 条件对象
        """
        condition = get_condition_from_pool()
        condition.equals('event_type', UI_EVENT_FEATURE_TOGGLE)
        condition.AND()
        condition.equals('feature_name', feature_name)
        condition.AND()
        condition.equals('state', state)
        return condition

    @staticmethod
    def element_type(element_type):
        """
        创建检查元素类型的条件

        Args:
            element_type: 元素类型（如按钮、表单等）

        Returns:
            Condition: 条件对象
        """
        return get_condition_from_pool().element_type_is(element_type)

    @staticmethod
    def element_in_group(group_name):
        """
        创建检查元素是否属于指定组的条件

        Args:
            group_name: 元素组名称

        Returns:
            Condition: 条件对象
        """
        return get_condition_from_pool().equals('element_group', group_name)

    @staticmethod
    def drag_event(source_element=None, target_element=None):
        """
        创建检查拖动事件的条件

        Args:
            source_element: 源元素ID
            target_element: 目标元素ID

        Returns:
            Condition: 条件对象
        """
        condition = get_condition_from_pool()
        condition.equals('event_type', UI_EVENT_DRAG)

        if source_element:
            condition.AND()
            condition.equals('source_element', source_element)

        if target_element:
            condition.AND()
            condition.equals('target_element', target_element)

        return condition
# ------ 第8部分结束 ------
# ------ 第9部分开始：简化API和单例函数 ------
# 简化API函数
def where(field):
    """
    创建条件构建器

    示例:
        where('user.age').greater_than(18).AND().equals('user.status', 'active')

    Args:
        field: 初始字段名

    Returns:
        FieldConditionBuilder: 字段条件构建器
    """
    return FieldConditionBuilder(field)


# 便捷函数 - 预配置的UI事件条件
def on_click(element_id=None):
    """
    创建检查点击事件的条件

    Args:
        element_id: 可选的元素ID

    Returns:
        Condition: 条件对象
    """
    condition = get_condition_from_pool()
    condition.equals('event_type', UI_EVENT_CLICK)

    if element_id:
        condition.AND()
        condition.element_is(element_id)

    return condition


def on_hover(element_id=None):
    """
    创建检查悬停事件的条件

    Args:
        element_id: 可选的元素ID

    Returns:
        Condition: 条件对象
    """
    condition = get_condition_from_pool()
    condition.equals('event_type', UI_EVENT_HOVER)

    if element_id:
        condition.AND()
        condition.element_is(element_id)

    return condition


def on_key(key=None):
    """
    创建检查按键事件的条件

    Args:
        key: 可选的键值

    Returns:
        Condition: 条件对象
    """
    condition = get_condition_from_pool()
    condition.equals('event_type', UI_EVENT_KEY_PRESS)

    if key:
        condition.AND()
        condition.key_is(key)

    return condition


def on_drag(source=None, target=None):
    """
    创建检查拖动事件的条件

    Args:
        source: 可选的源元素ID
        target: 可选的目标元素ID

    Returns:
        Condition: 条件对象
    """
    condition = get_condition_from_pool()
    condition.equals('event_type', UI_EVENT_DRAG)

    if source:
        condition.AND()
        condition.equals('source_element', source)

    if target:
        condition.AND()
        condition.equals('target_element', target)

    return condition


def on_feature_toggle(feature_name=None):
    """
    创建检查功能切换事件的条件

    Args:
        feature_name: 可选的功能名称

    Returns:
        Condition: 条件对象
    """
    condition = get_condition_from_pool()
    condition.equals('event_type', UI_EVENT_FEATURE_TOGGLE)

    if feature_name:
        condition.AND()
        condition.equals('feature_name', feature_name)

    return condition


def element_of_type(element_type):
    """
    创建检查元素类型的条件

    Args:
        element_type: 元素类型

    Returns:
        Condition: 条件对象
    """
    return get_condition_from_pool().element_type_is(element_type)


def in_element_group(group_name):
    """
    创建检查元素组的条件

    Args:
        group_name: 元素组名称

    Returns:
        Condition: 条件对象
    """
    return get_condition_from_pool().equals('element_group', group_name)


# 性能优化的UI事件条件工厂
class UIEventConditions:
    """
    UI事件条件工厂 - 提供预优化的UI事件条件

    内部缓存常用条件对象，避免重复创建
    """
    # 缓存常用条件
    _click_condition = None
    _hover_condition = None
    _key_press_condition = None
    _drag_condition = None

    @classmethod
    def click(cls):
        """获取基本点击条件"""
        if cls._click_condition is None:
            cls._click_condition = Condition().equals('event_type',
                                                      UI_EVENT_CLICK)
        return cls._click_condition

    @classmethod
    def hover(cls):
        """获取基本悬停条件"""
        if cls._hover_condition is None:
            cls._hover_condition = Condition().equals('event_type',
                                                      UI_EVENT_HOVER)
        return cls._hover_condition

    @classmethod
    def key_press(cls):
        """获取基本按键条件"""
        if cls._key_press_condition is None:
            cls._key_press_condition = Condition().equals('event_type',
                                                          UI_EVENT_KEY_PRESS)
        return cls._key_press_condition

    @classmethod
    def drag(cls):
        """获取基本拖动条件"""
        if cls._drag_condition is None:
            cls._drag_condition = Condition().equals('event_type',
                                                     UI_EVENT_DRAG)
        return cls._drag_condition

    @classmethod
    def click_on(cls, element_id):
        """获取点击特定元素的条件"""
        # 这些不缓存，因为元素ID会变化
        return Condition().equals('event_type',
                                  UI_EVENT_CLICK).AND().element_is(element_id)

    @classmethod
    def hover_on(cls, element_id):
        """获取悬停在特定元素上的条件"""
        return Condition().equals('event_type',
                                  UI_EVENT_HOVER).AND().element_is(element_id)


# 条件性事件系统单例
_conditional_event_system = None


def get_conditional_event_system(base_event_system=None):
    """
    获取或创建条件性事件系统实例

    Args:
        base_event_system: 基础事件系统

    Returns:
        ConditionalEventSystem: 条件性事件系统实例
    """
    global _conditional_event_system

    if _conditional_event_system is None:
        # 如果没有提供基础事件系统，尝试从现有系统获取
        if base_event_system is None:
            try:
                from utils.event_system import get_event_system
                base_event_system = get_event_system()
            except (ImportError, Exception):
                # 创建新实例
                logger.warning("无法获取全局事件系统，创建新实例")
                base_event_system = EnhancedEventSystem()

        _conditional_event_system = ConditionalEventSystem(base_event_system)

    return _conditional_event_system


# 如果直接运行此模块，执行测试代码
if __name__ == "__main__":
    # 创建条件性事件系统
    events = ConditionalEventSystem()

    # 配置优化选项
    events.set_batch_processing(True, batch_size=5, batch_interval=0.05)

    print("=" * 50)
    print("条件性事件系统测试")
    print("=" * 50)

    # 示例1: 使用条件对象进行订阅
    print("\n1. 基本条件订阅测试")
    condition = Condition().equals('status', 'error').AND().greater_than(
        'importance', 5)


    def error_handler(data):
        print(f"处理高重要性错误: {data}")


    events.subscribe_if("system_event", condition, error_handler)


    # 示例2: 使用where简化API进行订阅
    def notification_handler(data):
        print(f"发送通知: {data['message']}")


    notification_condition = where('type').equals('notification').AND().exists(
        'message')
    events.subscribe_if("user_event", notification_condition,
                        notification_handler)


    # 示例3: 使用链式API进行订阅
    def premium_user_handler(data):
        print(f"处理高级用户事件: {data}")


    events.subscribe_when(
        lambda: where('user.type').equals('premium').AND().greater_than(
            'user.level', 3)
    ).then("user_action", premium_user_handler)


    # 示例4: 使用UI事件特定API
    def button_click_handler(data):
        print(f"按钮点击: {data}")


    # 订阅特定按钮的点击事件
    events.subscribe_if(UI_EVENT_CLICK, on_click("submit_button"),
                        button_click_handler)


    # 使用区域条件
    def area_click_handler(data):
        print(f"区域点击: {data}")


    events.subscribe_if(
        UI_EVENT_CLICK,
        UIConditionBuilder.click_in_area(100, 100, 300, 200),
        area_click_handler
    )


    # 使用键盘事件
    def escape_key_handler(data):
        print(f"ESC键按下: {data}")


    events.subscribe_if(
        UI_EVENT_KEY_PRESS,
        on_key("Escape"),
        escape_key_handler
    )

    # 发布事件，测试条件匹配
    print("\n2. 测试条件匹配:")

    # 应该匹配error_handler
    print("\n发布匹配error_handler的事件:")
    events.publish("system_event", {
        'status': 'error',
        'importance': 8,
        'message': '系统崩溃'
    })

    # 不应该匹配error_handler (importance太低)
    print("\n发布不匹配error_handler的事件:")
    events.publish("system_event", {
        'status': 'error',
        'importance': 3,
        'message': '次要错误'
    })

    # 应该匹配notification_handler
    print("\n发布匹配notification_handler的事件:")
    events.publish("user_event", {
        'type': 'notification',
        'message': '您有一条新消息',
        'timestamp': time.time()
    })

    # 应该匹配premium_user_handler
    print("\n发布匹配premium_user_handler的事件:")
    events.publish("user_action", {
        'action': 'purchase',
        'user': {
            'type': 'premium',
            'level': 5,
            'name': 'Alice'
        }
    })

    # 不应该匹配premium_user_handler (level太低)
    print("\n发布不匹配premium_user_handler的事件:")
    events.publish("user_action", {
        'action': 'login',
        'user': {
            'type': 'premium',
            'level': 2,
            'name': 'Bob'
        }
    })

    # 测试UI事件条件
    print("\n发布UI点击事件:")
    events.publish(UI_EVENT_CLICK, {
        'element': 'submit_button',
        'x': 150,
        'y': 150,
        'timestamp': time.time()
    })

    # 测试按键事件
    print("\n发布按键事件:")
    events.publish(UI_EVENT_KEY_PRESS, {
        'key': 'Escape',
        'timestamp': time.time()
    })

    # 等待批处理完成
    time.sleep(0.1)

    # 显示系统统计
    print("\n3. 系统性能统计:")
    stats = events.get_stats()

    print("\n订阅统计:")
    for key, value in stats['subscriptions'].items():
        print(f"  {key}: {value}")

    print("\n评估统计:")
    for key, value in stats['evaluations'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n缓存统计:")
    for key, value in stats['cache'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n对象池统计:")
    for key, value in stats['pool'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # 测试条件缓存性能
    print("\n4. 测试条件缓存性能:")
    # 创建使用同一数据的多次评估条件
    test_condition = Condition().equals('element', 'test').AND().in_area(
        'position', 10, 10, 100, 100)

    # 创建测试数据
    test_data = {
        'event_type': UI_EVENT_CLICK,
        'element': 'test',
        'x': 50,
        'y': 50,
        'timestamp': time.time()
    }

    # 预热缓存
    test_condition.evaluate(test_data)

    # 评估多次 - 缓存模式
    iterations = 10000
    start_time = time.time()
    for _ in range(iterations):
        test_condition.evaluate(test_data)
    elapsed = time.time() - start_time

    print(f"  缓存模式: {iterations}次评估用时: {elapsed:.4f}秒")
    print(f"  每次评估平均时间: {elapsed / iterations * 1000:.4f}毫秒")

    # 禁用缓存重新测试
    test_condition.disable_cache()

    # 评估多次 - 非缓存模式
    start_time = time.time()
    for _ in range(iterations):
        test_condition.evaluate(test_data)
    elapsed = time.time() - start_time

    print(f"  非缓存模式: {iterations}次评估用时: {elapsed:.4f}秒")
    print(f"  每次评估平均时间: {elapsed / iterations * 1000:.4f}毫秒")

    # 批处理性能测试
    print("\n5. 批处理性能测试:")

    # 创建一些UI事件
    ui_events = []
    for i in range(100):
        ui_events.append({
            'element': f'button_{i % 10}',
            'x': i * 2,
            'y': i % 50,
            'timestamp': time.time()
        })

    # 测试批处理模式
    events.set_batch_processing(True, batch_size=10, batch_interval=0.02)

    start_time = time.time()
    for data in ui_events:
        events.publish(UI_EVENT_CLICK, data)
    elapsed = time.time() - start_time

    print(f"  批处理模式: 发布100个UI事件用时: {elapsed:.4f}秒")
    print(f"  每个事件平均时间: {elapsed / 100 * 1000:.4f}毫秒")

    # 等待批处理完成
    time.sleep(0.1)

    # 测试非批处理模式
    events.set_batch_processing(False)

    start_time = time.time()
    for data in ui_events:
        events.publish(UI_EVENT_CLICK, data)
    elapsed = time.time() - start_time

    print(f"  非批处理模式: 发布100个UI事件用时: {elapsed:.4f}秒")
    print(f"  每个事件平均时间: {elapsed / 100 * 1000:.4f}毫秒")

    # 条件销毁和资源回收测试
    print("\n6. 资源管理测试:")

    # 创建多个条件
    conditions = []
    for i in range(1000):
        c = Condition().equals('value', i)
        conditions.append(c)

    print(f"  创建了1000个条件对象")

    # 销毁一半条件
    for i in range(500):
        conditions[i].destroy()

    # 更新条件引用
    conditions = conditions[500:]

    print(f"  销毁了500个条件对象")

    # 收集垃圾
    collected = collect_garbage()
    print(f"  垃圾回收: 回收了 {collected} 个对象")

    # 获取对象池统计
    pool_stats = get_pool_stats()
    print(f"  对象池大小: {pool_stats['size']}/{pool_stats['max_size']}")

    # 清理资源并结束
    conditions = None
    events.shutdown()
    print("\n测试完成，资源已清理")
# ------ 第9部分结束 ------

