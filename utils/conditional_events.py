# -*- coding: utf-8 -*-
"""
条件性事件订阅功能 - 事件系统的扩展模块

本模块扩展了现有事件系统，提供:
1. 基于条件表达式的事件订阅
2. 属性匹配过滤器
3. 复合条件构建器
4. 链式API支持

使用此功能，您可以通过简单的语法轻松指定事件过滤条件。
"""
import time
import logging
from functools import partial

# 导入基础事件系统，兼容不同导入路径
try:
    from utils.event_system import EnhancedEventSystem
except ImportError:
    try:
        from event_system import EnhancedEventSystem
    except ImportError:
        # 在无法导入时提供一个基本接口，以便代码能够编译
        class EnhancedEventSystem:
            def __init__(self): pass

            def subscribe(self, event_type, handler): pass

            def subscribe_with_filter(self, event_type, handler,
                                      filter_func): pass

            def subscribe_with_priority(self, event_type, handler, priority,
                                        filter_func): pass

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


class Condition:
    """
    条件表达式构建器，用于创建事件过滤条件

    提供直观的API来构建复合条件表达式，支持链式调用。
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

        if field is not None:
            self.conditions.append({
                'field': field,
                'operator': operator,
                'value': value,
                'join': None
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

        result = self._evaluate_condition(self.conditions[0], data)

        for i in range(1, len(self.conditions)):
            condition = self.conditions[i]
            cond_result = self._evaluate_condition(condition, data)

            if condition['join'] == 'AND':
                result = result and cond_result
            elif condition['join'] == 'OR':
                result = result or cond_result
            else:
                # 默认为AND
                result = result and cond_result

        return result

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

        # 检查嵌套字段路径 (如 "user.name")
        if '.' in field:
            parts = field.split('.')
            curr = data
            for part in parts:
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return False
            actual_value = curr
        else:
            # 简单字段
            if field not in data:
                return False
            actual_value = data[field]

        # 执行比较
        if operator == '==':
            return actual_value == expected_value
        elif operator == '!=':
            return actual_value != expected_value
        elif operator == '>':
            return actual_value > expected_value
        elif operator == '>=':
            return actual_value >= expected_value
        elif operator == '<':
            return actual_value < expected_value
        elif operator == '<=':
            return actual_value <= expected_value
        elif operator == 'in':
            return actual_value in expected_value
        elif operator == 'not_in':
            return actual_value not in expected_value
        elif operator == 'contains':
            return expected_value in actual_value
        elif operator == 'starts_with':
            return str(actual_value).startswith(str(expected_value))
        elif operator == 'ends_with':
            return str(actual_value).endswith(str(expected_value))
        elif operator == 'matches':
            if callable(expected_value):
                return expected_value(actual_value)
            return False
        else:
            logger.warning(f"未知操作符: {operator}")
            return False

    def __call__(self, data):
        """
        使对象可调用，方便作为过滤函数

        Args:
            data: 要检查的数据

        Returns:
            bool: 条件是否匹配
        """
        return self.evaluate(data)

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
        return self._add_condition(field, 'matches', lambda x: x is not None)

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
            'join': self.current_join
        }

        self.conditions.append(condition)
        self.current_join = None
        return self


class ConditionalEventSystem:
    """
    条件性事件系统

    扩展EnhancedEventSystem，添加条件式订阅支持，
    使用更直观的API进行事件过滤。
    """

    def __init__(self, base_event_system=None):
        """
        初始化条件性事件系统

        Args:
            base_event_system: 基础事件系统实例，如果为None则创建新实例
        """
        # 使用提供的事件系统或创建新的
        self.event_system = base_event_system or EnhancedEventSystem()
        logger.info("条件性事件系统已初始化")

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
        if isinstance(condition, Condition):
            filter_func = condition
        elif callable(condition):
            filter_func = condition
        else:
            logger.error(f"无效的条件类型: {type(condition)}")
            return False

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
        return self.event_system.unsubscribe(event_type, handler)

    # 委托其他方法到基础事件系统
    def __getattr__(self, name):
        """转发未定义的方法到基础事件系统"""
        return getattr(self.event_system, name)


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
        self.condition = Condition()

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


# 获取条件性事件系统实例
_conditional_event_system = None


def get_conditional_event_system(base_event_system=None):
    """
    获取或创建条件性事件系统

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
                pass

        _conditional_event_system = ConditionalEventSystem(base_event_system)

    return _conditional_event_system


# 使用示例
if __name__ == "__main__":
    # 创建条件性事件系统
    events = ConditionalEventSystem()

    # 示例1: 使用条件对象进行订阅
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

    # 发布事件，测试条件匹配
    print("\n测试条件匹配:")

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
