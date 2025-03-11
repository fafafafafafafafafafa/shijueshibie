# -*- coding: utf-8 -*-
"""
UI事件系统 - 提供UI组件与事件系统的集成

本模块提供:
1. 专用的UI事件类型定义
2. UI组件与事件系统的集成接口
3. 事件驱动的UI更新机制
"""
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

    # 显示事件
    DISPLAY_UPDATED = "display_updated"
    FRAME_RENDERED = "frame_rendered"
    RENDER_COMPLETED = "render_completed"

    # 用户交互事件
    KEY_PRESSED = "key_pressed"
    MOUSE_CLICKED = "mouse_clicked"
    MOUSE_MOVED = "mouse_moved"

    # UI状态事件
    VIEW_MODE_CHANGED = "view_mode_changed"
    OPTION_TOGGLED = "option_toggled"
    FEATURE_TOGGLED = "feature_toggled"

    # 信息展示事件
    DEBUG_INFO_UPDATED = "debug_info_updated"
    STATUS_CHANGED = "status_changed"
    NOTIFICATION_SHOWN = "notification_shown"

    # 性能事件
    FPS_UPDATED = "fps_updated"
    PERFORMANCE_WARNING = "performance_warning"


class UIEventPublisher:
    """
    UI事件发布器 - 提供UI组件发布事件的统一接口

    此类作为UI组件和事件系统之间的桥梁，提供友好的API来发布UI相关事件。
    """

    def __init__(self, component_name=None):
        """
        初始化UI事件发布器

        Args:
            component_name: UI组件名称，用于标识事件源
        """
        self.component_name = component_name or "unknown_component"
        self.event_system = get_event_system()
        logger.info(f"UI事件发布器已初始化: {self.component_name}")

    def publish_window_event(self, event_type, window_name, window_size=None,
                             **kwargs):
        """
        发布窗口相关事件

        Args:
            event_type: 事件类型，如 UIEventTypes.WINDOW_CREATED
            window_name: 窗口名称
            window_size: 窗口尺寸 (width, height)
            **kwargs: 其他相关数据
        """
        data = {
            "component": self.component_name,
            "window_name": window_name,
            "timestamp": None,  # 由事件系统添加
            **kwargs
        }

        if window_size:
            data["window_size"] = window_size

        self.event_system.publish(event_type, data)

    def publish_display_event(self, event_type, display_data=None,
                              frame_info=None, **kwargs):
        """
        发布显示相关事件

        Args:
            event_type: 事件类型，如 UIEventTypes.DISPLAY_UPDATED
            display_data: 显示相关数据
            frame_info: 帧相关信息，如尺寸、类型等
            **kwargs: 其他相关数据
        """
        data = {
            "component": self.component_name,
            "timestamp": None,  # 由事件系统添加
            **kwargs
        }

        if display_data:
            data["display_data"] = display_data

        if frame_info:
            data["frame_info"] = frame_info

        self.event_system.publish(event_type, data)

    def publish_user_interaction(self, event_type, interaction_type, value,
                                 **kwargs):
        """
        发布用户交互事件

        Args:
            event_type: 事件类型，如 UIEventTypes.KEY_PRESSED
            interaction_type: 交互类型，如 "keyboard", "mouse"
            value: 交互值，如按键代码、鼠标位置等
            **kwargs: 其他相关数据
        """
        data = {
            "component": self.component_name,
            "interaction_type": interaction_type,
            "value": value,
            "timestamp": None,  # 由事件系统添加
            **kwargs
        }

        self.event_system.publish(event_type, data)

    def publish_ui_state_change(self, event_type, state_name, old_value,
                                new_value, **kwargs):
        """
        发布UI状态变更事件

        Args:
            event_type: 事件类型，如 UIEventTypes.OPTION_TOGGLED
            state_name: 状态名称
            old_value: 旧值
            new_value: 新值
            **kwargs: 其他相关数据
        """
        data = {
            "component": self.component_name,
            "state_name": state_name,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": None,  # 由事件系统添加
            **kwargs
        }

        self.event_system.publish(event_type, data)

    def publish_notification(self, message, level="info", duration=3.0,
                             **kwargs):
        """
        发布通知事件

        Args:
            message: 通知消息
            level: 通知级别 (info, warning, error)
            duration: 通知显示时长(秒)
            **kwargs: 其他相关数据
        """
        data = {
            "component": self.component_name,
            "message": message,
            "level": level,
            "duration": duration,
            "timestamp": None,  # 由事件系统添加
            **kwargs
        }

        self.event_system.publish(UIEventTypes.NOTIFICATION_SHOWN, data)

    def publish_fps_update(self, fps, frame_time=None, **kwargs):
        """
        发布FPS更新事件

        Args:
            fps: 当前帧率
            frame_time: 单帧处理时间(毫秒)
            **kwargs: 其他相关数据
        """
        data = {
            "component": self.component_name,
            "fps": fps,
            "timestamp": None,  # 由事件系统添加
            **kwargs
        }

        if frame_time:
            data["frame_time"] = frame_time

        self.event_system.publish(UIEventTypes.FPS_UPDATED, data)

        # 如果FPS过低，也发布性能警告事件
        if fps < 15:
            warning_data = {
                "component": self.component_name,
                "warning_type": "low_fps",
                "fps": fps,
                "timestamp": None,  # 由事件系统添加
                **kwargs
            }
            self.event_system.publish(UIEventTypes.PERFORMANCE_WARNING,
                                      warning_data)


class UIEventSubscriber:
    """
    UI事件订阅器 - 提供UI组件订阅事件的统一接口

    此类作为UI组件和事件系统之间的桥梁，提供友好的API来订阅和处理UI相关事件。
    """

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
        logger.info(f"UI事件订阅器已初始化: {self.component_name}")

    def subscribe(self, event_type, handler, priority=0):
        """
        订阅UI事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数
            priority: 处理优先级

        Returns:
            bool: 是否成功订阅
        """
        try:
            success = self.event_system.subscribe_with_priority(
                event_type, handler, priority)

            if success:
                # 记录订阅，用于后续取消订阅
                if event_type not in self.subscriptions:
                    self.subscriptions[event_type] = []
                self.subscriptions[event_type].append(handler)

                logger.debug(
                    f"UI组件 {self.component_name} 订阅了事件: {event_type}")

            return success
        except Exception as e:
            logger.error(
                f"UI组件 {self.component_name} 订阅事件 {event_type} 失败: {e}")
            return False

    def subscribe_conditional(self, event_type, condition_builder, handler,
                              priority=0):
        """
        条件性订阅UI事件

        Args:
            event_type: 事件类型
            condition_builder: 条件构建器函数或条件对象
            handler: 事件处理函数
            priority: 处理优先级

        Returns:
            bool: 是否成功订阅
        """
        try:
            success = self.conditional_events.subscribe_if(
                event_type, condition_builder, handler, priority)

            if success:
                # 记录订阅，用于后续取消订阅
                if event_type not in self.subscriptions:
                    self.subscriptions[event_type] = []
                self.subscriptions[event_type].append(handler)

                logger.debug(
                    f"UI组件 {self.component_name} 条件性订阅了事件: {event_type}")

            return success
        except Exception as e:
            logger.error(
                f"UI组件 {self.component_name} 条件性订阅事件 {event_type} 失败: {e}")
            return False

    def subscribe_to_notifications(self, handler, level=None, component=None,
                                   priority=0):
        """
        订阅通知事件

        Args:
            handler: 通知处理函数
            level: 可选的通知级别过滤 (info, warning, error)
            component: 可选的组件名称过滤
            priority: 处理优先级

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
            success = self.conditional_events.subscribe_if(
                UIEventTypes.NOTIFICATION_SHOWN, condition, handler, priority)

            if success:
                # 记录订阅
                if UIEventTypes.NOTIFICATION_SHOWN not in self.subscriptions:
                    self.subscriptions[UIEventTypes.NOTIFICATION_SHOWN] = []
                self.subscriptions[UIEventTypes.NOTIFICATION_SHOWN].append(
                    handler)

                logger.debug(f"UI组件 {self.component_name} 订阅了通知事件")

            return success
        except Exception as e:
            logger.error(f"UI组件 {self.component_name} 订阅通知事件失败: {e}")
            return False

    def unsubscribe_all(self):
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

        logger.info(f"UI组件 {self.component_name} 取消了 {count} 个事件订阅")
        return count

    def unsubscribe(self, event_type, handler=None):
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
            # 取消特定处理函数
            try:
                if self.event_system.unsubscribe(event_type, handler):
                    self.subscriptions[event_type].remove(handler)
                    count = 1
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
                except Exception as e:
                    logger.error(
                        f"取消订阅事件 {event_type} 的处理函数失败: {e}")

            # 清空该事件类型的记录
            self.subscriptions[event_type] = []

        logger.info(
            f"UI组件 {self.component_name} 取消了 {count} 个 {event_type} 事件订阅")
        return count


class UIEventDrivenComponent:
    """
    事件驱动UI组件基类 - 提供事件驱动UI组件的基础实现

    此类作为事件驱动UI组件的基类，提供与事件系统集成的基础功能。
    派生类可以覆盖方法来实现自定义的UI更新行为。
    """

    def __init__(self, component_name):
        """
        初始化事件驱动UI组件

        Args:
            component_name: 组件名称
        """
        self.component_name = component_name
        self.publisher = UIEventPublisher(component_name)
        self.subscriber = UIEventSubscriber(component_name)
        self._setup_event_handlers()
        logger.info(f"事件驱动UI组件已初始化: {component_name}")

    def _setup_event_handlers(self):
        """
        设置事件处理器

        派生类应覆盖此方法来设置所需的事件处理器。
        """
        pass

    def handle_notification(self, data):
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

    def handle_performance_warning(self, data):
        """
        处理性能警告事件

        Args:
            data: 性能警告事件数据
        """
        # 基本实现，派生类可以覆盖
        warning_type = data.get('warning_type', 'unknown')
        logger.warning(
            f"UI组件 {self.component_name} 收到性能警告: {warning_type}")

    def handle_ui_state_change(self, data):
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

    def update(self, force=False):
        """
        更新UI组件

        Args:
            force: 是否强制更新

        Returns:
            bool: 是否成功更新
        """
        # 基本实现，派生类应覆盖此方法
        return True

    def cleanup(self):
        """
        清理资源，取消事件订阅

        Returns:
            bool: 是否成功清理
        """
        try:
            self.subscriber.unsubscribe_all()
            logger.info(f"UI组件 {self.component_name} 已清理资源")
            return True
        except Exception as e:
            logger.error(f"UI组件 {self.component_name} 清理资源失败: {e}")
            return False


# 辅助函数和工具

def create_ui_condition(component=None, **kwargs):
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
