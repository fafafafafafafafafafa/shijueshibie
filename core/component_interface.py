# -*- coding: utf-8 -*-
"""
组件接口定义模块 - 提供基础组件接口和相关协议
定义组件生命周期方法、事件处理和配置管理接口
"""

import threading
import queue
import time
from typing import Tuple
import abc
import logging
from typing import Dict, List, Set, Any, Optional, Callable, Union, Type
from enum import Enum

# 导入核心组件
from core.component_lifecycle import LifecycleState, LifecycleManager
from core.event_models import Event, EventType

logger = logging.getLogger("ComponentInterface")


class ComponentInterface(abc.ABC):
    """
    基础组件接口 - 所有组件的基础接口
    定义组件的基本属性和方法
    """

    @abc.abstractmethod
    def get_component_id(self) -> str:
        """
        获取组件ID

        Returns:
            str: 组件ID
        """
        pass

    @abc.abstractmethod
    def get_component_type(self) -> str:
        """
        获取组件类型

        Returns:
            str: 组件类型名称
        """
        pass

    @abc.abstractmethod
    def get_component_info(self) -> Dict[str, Any]:
        """
        获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        pass

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取组件统计信息

        Returns:
            Dict[str, Any]: 组件统计信息
        """
        pass


class LifecycleComponentInterface(ComponentInterface):
    """
    生命周期组件接口 - 支持生命周期管理的组件接口
    定义组件的生命周期方法
    """

    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        初始化组件

        Returns:
            bool: 是否成功初始化
        """
        pass

    @abc.abstractmethod
    def start(self) -> bool:
        """
        启动组件

        Returns:
            bool: 是否成功启动
        """
        pass

    @abc.abstractmethod
    def pause(self) -> bool:
        """
        暂停组件

        Returns:
            bool: 是否成功暂停
        """
        pass

    @abc.abstractmethod
    def resume(self) -> bool:
        """
        恢复组件

        Returns:
            bool: 是否成功恢复
        """
        pass

    @abc.abstractmethod
    def stop(self) -> bool:
        """
        停止组件

        Returns:
            bool: 是否成功停止
        """
        pass

    @abc.abstractmethod
    def destroy(self) -> bool:
        """
        销毁组件

        Returns:
            bool: 是否成功销毁
        """
        pass

    @abc.abstractmethod
    def get_state(self) -> LifecycleState:
        """
        获取组件当前状态

        Returns:
            LifecycleState: 组件生命周期状态
        """
        pass

    @abc.abstractmethod
    def is_running(self) -> bool:
        """
        检查组件是否运行中

        Returns:
            bool: 是否运行中
        """
        pass


class EventAwareComponentInterface(ComponentInterface):
    """
    事件感知组件接口 - 支持事件处理的组件接口
    定义组件的事件处理方法
    """

    @abc.abstractmethod
    def handle_event(self, event: Event) -> bool:
        """
        处理事件

        Args:
            event: 事件对象

        Returns:
            bool: 是否成功处理事件
        """
        pass

    @abc.abstractmethod
    def get_handled_event_types(self) -> Set[EventType]:
        """
        获取组件处理的事件类型

        Returns:
            Set[EventType]: 处理的事件类型集合
        """
        pass

    @abc.abstractmethod
    def subscribe_to_events(self) -> bool:
        """
        订阅事件

        Returns:
            bool: 是否成功订阅
        """
        pass

    @abc.abstractmethod
    def unsubscribe_from_events(self) -> bool:
        """
        取消订阅事件

        Returns:
            bool: 是否成功取消订阅
        """
        pass


class ConfigurableComponentInterface(ComponentInterface):
    """
    可配置组件接口 - 支持配置管理的组件接口
    定义组件的配置管理方法
    """

    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        获取组件配置

        Returns:
            Dict[str, Any]: 组件配置
        """
        pass

    @abc.abstractmethod
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            config: 新配置

        Returns:
            bool: 是否成功更新配置
        """
        pass

    @abc.abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取组件配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        pass

    @abc.abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[
        bool, Optional[str]]:
        """
        验证配置是否有效

        Args:
            config: 要验证的配置

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        pass


class ResourceAwareComponentInterface(ComponentInterface):
    """
    资源感知组件接口 - 支持资源管理的组件接口
    定义组件的资源管理方法
    """

    @abc.abstractmethod
    def get_resource_requirements(self) -> Dict[str, Any]:
        """
        获取组件资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        pass

    @abc.abstractmethod
    def adapt_to_resources(self, available_resources: Dict[str, Any]) -> bool:
        """
        适应可用资源

        Args:
            available_resources: 可用资源

        Returns:
            bool: 是否成功适应
        """
        pass

    @abc.abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取组件资源使用情况

        Returns:
            Dict[str, Any]: 资源使用情况
        """
        pass


class BaseComponent(ComponentInterface):
    """
    基础组件实现 - 提供组件接口的基本实现
    实现组件的通用功能
    """

    def __init__(self, component_id: str, component_type: Optional[str] = None):
        """
        初始化基础组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
        """
        self._component_id = component_id
        self._component_type = component_type or self.__class__.__name__
        self._stats = {
            'operations_count': 0,
            'error_count': 0,
            'last_operation_time': 0.0,
            'total_processing_time': 0.0
        }

    def get_component_id(self) -> str:
        """
        获取组件ID

        Returns:
            str: 组件ID
        """
        return self._component_id

    def get_component_type(self) -> str:
        """
        获取组件类型

        Returns:
            str: 组件类型名称
        """
        return self._component_type

    def get_component_info(self) -> Dict[str, Any]:
        """
        获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        return {
            'component_id': self._component_id,
            'component_type': self._component_type,
            'stats': self.get_stats()
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取组件统计信息

        Returns:
            Dict[str, Any]: 组件统计信息
        """
        return dict(self._stats)

    def update_stats(self, operation_type: str, processing_time: float,
                     error: bool = False):
        """
        更新组件统计信息

        Args:
            operation_type: 操作类型
            processing_time: 处理时间
            error: 是否出错
        """
        import time

        self._stats['operations_count'] += 1
        self._stats['last_operation_time'] = time.time()
        self._stats['total_processing_time'] += processing_time

        if error:
            self._stats['error_count'] += 1

        # 更新特定操作类型的统计
        op_key = f'{operation_type}_count'
        if op_key in self._stats:
            self._stats[op_key] += 1
        else:
            self._stats[op_key] = 1


class BaseLifecycleComponent(BaseComponent, LifecycleComponentInterface):
    """
    基础生命周期组件 - 提供生命周期组件接口的基本实现
    实现组件的生命周期管理功能
    """

    def __init__(self, component_id: str, component_type: Optional[str] = None):
        """
        初始化基础生命周期组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
        """
        super().__init__(component_id, component_type)

        # 创建生命周期管理器
        from core.component_lifecycle import LifecycleManager
        self._lifecycle_manager = LifecycleManager(component_id,
                                                   self._component_type,
                                                   initial_state=LifecycleState.REGISTERED)  # 设置初始状态为 REGISTERED)

        # 添加锁来保护并发操作
        self._lifecycle_lock = threading.RLock()

    def initialize(self) -> bool:
        """
        初始化组件

        Returns:
            bool: 是否成功初始化
        """
        with self._lifecycle_lock:
            try:
                # 先转换到 INITIALIZING 状态
                if not self._lifecycle_manager.transition_to(
                        LifecycleState.INITIALIZING):
                    logger.warning(
                        f"无法转换到 INITIALIZING 状态: {self._component_id}")
                    return False

                # 执行初始化逻辑
                result = self._do_initialize()

                # 如果初始化成功，转换到 INITIALIZED 状态
                if result:
                    self._lifecycle_manager.transition_to(
                        LifecycleState.INITIALIZED)
                else:
                    self._lifecycle_manager.transition_to(LifecycleState.ERROR)

                return result
            except Exception as e:
                logger.error(f"组件初始化错误: {self._component_id}, 错误: {e}")
                self._lifecycle_manager.transition_to(LifecycleState.ERROR)
                return False

    def start(self) -> bool:
        """
        启动组件

        Returns:
            bool: 是否成功启动
        """
        try:
            # 检查是否可以启动
            current_state = self.get_state()
            if current_state not in [LifecycleState.INITIALIZED,
                                     LifecycleState.STOPPED,
                                     LifecycleState.PAUSED]:
                logger.warning(
                    f"组件无法从 {current_state.name} 状态启动: {self._component_id}")
                return False

            # 先转换到 STARTING 状态
            if not self._lifecycle_manager.transition_to(
                    LifecycleState.STARTING):
                logger.warning(
                    f"无法转换到 STARTING 状态: {self._component_id}")
                return False

            # 执行启动逻辑
            result = self._do_start()

            # 如果启动成功，转换到 RUNNING 状态
            if result:
                self._lifecycle_manager.transition_to(LifecycleState.RUNNING)
            else:
                self._lifecycle_manager.transition_to(LifecycleState.ERROR)

            return result
        except Exception as e:
            logger.error(f"组件启动错误: {self._component_id}, 错误: {e}")
            self._lifecycle_manager.transition_to(LifecycleState.ERROR)
            return False

    def pause(self) -> bool:
        """
        暂停组件

        Returns:
            bool: 是否成功暂停
        """
        try:
            # 检查是否可以暂停
            if self.get_state() != LifecycleState.RUNNING:
                logger.warning(f"组件未运行，无法暂停: {self._component_id}")
                return False

            # 先转换到 PAUSING 状态
            if not self._lifecycle_manager.transition_to(
                    LifecycleState.PAUSING):
                logger.warning(f"无法转换到 PAUSING 状态: {self._component_id}")
                return False

            # 执行暂停逻辑
            result = self._do_pause()

            # 如果暂停成功，转换到 PAUSED 状态
            if result:
                self._lifecycle_manager.transition_to(LifecycleState.PAUSED)
            else:
                self._lifecycle_manager.transition_to(LifecycleState.ERROR)

            return result
        except Exception as e:
            logger.error(f"组件暂停错误: {self._component_id}, 错误: {e}")
            self._lifecycle_manager.transition_to(LifecycleState.ERROR)
            return False

    def resume(self) -> bool:
        """
        恢复组件

        Returns:
            bool: 是否成功恢复
        """
        try:
            # 检查是否可以恢复
            if self.get_state() != LifecycleState.PAUSED:
                logger.warning(f"组件未暂停，无法恢复: {self._component_id}")
                return False

            # 先转换到 STARTING 状态
            if not self._lifecycle_manager.transition_to(
                    LifecycleState.STARTING):
                logger.warning(
                    f"无法转换到 STARTING 状态: {self._component_id}")
                return False

            # 执行恢复逻辑
            result = self._do_resume()

            # 如果恢复成功，转换到 RUNNING 状态
            if result:
                self._lifecycle_manager.transition_to(LifecycleState.RUNNING)
            else:
                self._lifecycle_manager.transition_to(LifecycleState.ERROR)

            return result
        except Exception as e:
            logger.error(f"组件恢复错误: {self._component_id}, 错误: {e}")
            self._lifecycle_manager.transition_to(LifecycleState.ERROR)
            return False

    def stop(self) -> bool:
        """
        停止组件

        Returns:
            bool: 是否成功停止
        """
        try:
            # 检查是否可以停止
            current_state = self.get_state()
            if current_state not in [LifecycleState.RUNNING,
                                     LifecycleState.PAUSED,
                                     LifecycleState.ERROR]:
                logger.warning(
                    f"组件无法从 {current_state.name} 状态停止: {self._component_id}")
                return False

            # 转换到 STOPPING 状态
            if not self._lifecycle_manager.transition_to(
                    LifecycleState.STOPPING):
                logger.warning(
                    f"无法转换到 STOPPING 状态: {self._component_id}")
                return False

            # 执行停止逻辑
            result = self._do_stop()

            # 如果停止成功，转换到 STOPPED 状态
            if result:
                self._lifecycle_manager.transition_to(LifecycleState.STOPPED)
            else:
                self._lifecycle_manager.transition_to(LifecycleState.ERROR)

            return result
        except Exception as e:
            logger.error(f"组件停止错误: {self._component_id}, 错误: {e}")
            self._lifecycle_manager.transition_to(LifecycleState.ERROR)
            return False

    def destroy(self) -> bool:
        """
        销毁组件

        Returns:
            bool: 是否成功销毁
        """
        try:
            # 检查是否可以销毁
            current_state = self.get_state()
            if current_state not in [LifecycleState.INITIALIZED,
                                     LifecycleState.STOPPED]:
                logger.warning(
                    f"组件无法从 {current_state.name} 状态销毁: {self._component_id}")
                return False

            # 先转换到 DESTROYING 状态
            if not self._lifecycle_manager.transition_to(
                    LifecycleState.DESTROYING):
                logger.warning(
                    f"无法转换到 DESTROYING 状态: {self._component_id}")
                return False

            # 执行销毁逻辑
            result = self._do_destroy()

            # 如果销毁成功，转换到 DESTROYED 状态
            if result:
                self._lifecycle_manager.transition_to(LifecycleState.DESTROYED)
            else:
                self._lifecycle_manager.transition_to(LifecycleState.ERROR)

            return result
        except Exception as e:
            logger.error(f"组件销毁错误: {self._component_id}, 错误: {e}")
            self._lifecycle_manager.transition_to(LifecycleState.ERROR)
            return False

    def get_state(self) -> LifecycleState:
        """
        获取组件当前状态

        Returns:
            LifecycleState: 组件生命周期状态
        """
        return self._lifecycle_manager.get_current_state()

    def is_running(self) -> bool:
        """
        检查组件是否运行中

        Returns:
            bool: 是否运行中
        """
        return self.get_state() == LifecycleState.RUNNING

    def get_component_info(self) -> Dict[str, Any]:
        """
        获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        info = super().get_component_info()
        info['state'] = self.get_state().name
        info['uptime'] = self._lifecycle_manager.get_total_uptime()
        return info

    # 子类需要实现的方法

    def _do_initialize(self) -> bool:
        """
        执行初始化逻辑

        Returns:
            bool: 是否成功初始化
        """
        return True

    def _do_start(self) -> bool:
        """
        执行启动逻辑

        Returns:
            bool: 是否成功启动
        """
        return True

    def _do_pause(self) -> bool:
        """
        执行暂停逻辑

        Returns:
            bool: 是否成功暂停
        """
        return True

    def _do_resume(self) -> bool:
        """
        执行恢复逻辑

        Returns:
            bool: 是否成功恢复
        """
        return True

    def _do_stop(self) -> bool:
        """
        执行停止逻辑

        Returns:
            bool: 是否成功停止
        """
        return True

    def _do_destroy(self) -> bool:
        """
        执行销毁逻辑

        Returns:
            bool: 是否成功销毁
        """
        return True


class BaseEventAwareComponent(BaseComponent, EventAwareComponentInterface):
    """
    基础事件感知组件 - 提供事件感知组件接口的基本实现
    实现组件的事件处理功能
    """

    def __init__(self, component_id: str, component_type: Optional[str] = None):
        """
        初始化基础事件感知组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
        """
        super().__init__(component_id, component_type)

        # 获取事件总线
        from core.event_bus import get_event_bus
        self._event_bus = get_event_bus()

        # 初始化事件处理器映射 {事件类型: 处理函数}
        self._event_handlers = {}

        # 订阅者ID
        self._subscriber_id = None

    def handle_event(self, event: Event) -> bool:
        """
        处理事件

        Args:
            event: 事件对象

        Returns:
            bool: 是否成功处理事件
        """
        try:
            # 检查是否有对应的处理器
            if event.event_type in self._event_handlers:
                handler = self._event_handlers[event.event_type]
                handler(event)
                return True
            return False
        except Exception as e:
            logger.error(
                f"事件处理错误: {self._component_id}, 事件: {event.event_type.name}, 错误: {e}")
            return False

    def get_handled_event_types(self) -> Set[EventType]:
        """
        获取组件处理的事件类型

        Returns:
            Set[EventType]: 处理的事件类型集合
        """
        return set(self._event_handlers.keys())

    def subscribe_to_events(self) -> bool:
        """
        订阅事件

        Returns:
            bool: 是否成功订阅
        """
        try:
            # 取消现有订阅
            self.unsubscribe_from_events()

            # 获取处理的事件类型
            event_types = self.get_handled_event_types()
            if not event_types:
                return True  # 没有需要处理的事件类型

            # 创建事件过滤器
            from core.event_bus import EventFilter
            filter_ = EventFilter(event_types=event_types)

            # 订阅事件
            self._subscriber_id = self._event_bus.subscribe(
                event_type=None,  # 所有事件类型
                handler=self.handle_event,
                subscriber_id=f"{self._component_id}_subscriber",
                filter_=filter_
            )

            logger.debug(
                f"组件已订阅事件: {self._component_id}, 事件类型: {[t.name for t in event_types]}")
            return True
        except Exception as e:
            logger.error(f"事件订阅错误: {self._component_id}, 错误: {e}")
            return False

    def unsubscribe_from_events(self) -> bool:
        """
        取消订阅事件

        Returns:
            bool: 是否成功取消订阅
        """
        try:
            if self._subscriber_id:
                self._event_bus.unsubscribe(self._subscriber_id)
                self._subscriber_id = None
                logger.debug(f"组件已取消订阅事件: {self._component_id}")
            return True
        except Exception as e:
            logger.error(f"取消事件订阅错误: {self._component_id}, 错误: {e}")
            return False

    def register_event_handler(self, event_type: EventType,
                               handler: Callable[[Event], None]):
        """
        注册事件处理器

        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        self._event_handlers[event_type] = handler

    def publish_event(self, event_type: EventType, data: Any = None) -> str:
        """
        发布事件

        Args:
            event_type: 事件类型
            data: 事件数据

        Returns:
            str: 事件ID
        """
        # 创建并发布事件
        return self._event_bus.create_and_publish(
            event_type=event_type,
            data=data,
            source_id=self._component_id
        )


class BaseConfigurableComponent(BaseComponent, ConfigurableComponentInterface):
    """
    基础可配置组件 - 提供可配置组件接口的基本实现
    实现组件的配置管理功能
    """

    def __init__(self, component_id: str, component_type: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化基础可配置组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
            config: 初始配置
        """
        super().__init__(component_id, component_type)

        # 初始化配置
        self._config = config or {}

        # 配置模式
        self._config_schema = self._create_config_schema()

    def get_config(self) -> Dict[str, Any]:
        """
        获取组件配置

        Returns:
            Dict[str, Any]: 组件配置
        """
        return dict(self._config)

    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            config: 新配置

        Returns:
            bool: 是否成功更新配置
        """
        try:
            # 验证配置
            valid, error = self.validate_config(config)
            if not valid:
                logger.warning(
                    f"配置验证失败: {self._component_id}, 错误: {error}")
                return False

            # 更新配置
            old_config = dict(self._config)
            self._config.update(config)

            # 应用配置
            result = self._apply_config(old_config, self._config)
            if not result:
                # 恢复旧配置
                self._config = old_config
                logger.warning(f"应用配置失败: {self._component_id}")
                return False

            logger.debug(f"组件配置已更新: {self._component_id}")
            return True
        except Exception as e:
            logger.error(f"更新配置错误: {self._component_id}, 错误: {e}")
            return False

    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取组件配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        return dict(self._config_schema)

    def validate_config(self, config: Dict[str, Any]) -> Tuple[
        bool, Optional[str]]:
        """
        验证配置是否有效

        Args:
            config: 要验证的配置

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        try:
            # 默认实现：检查必需字段
            schema = self.get_config_schema()

            for field_name, field_info in schema.items():
                if field_info.get('required',
                                  False) and field_name not in config:
                    return False, f"缺少必需字段: {field_name}"

            return True, None
        except Exception as e:
            return False, str(e)

    def _create_config_schema(self) -> Dict[str, Any]:
        """
        创建配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        # 子类应重写此方法
        return {}

    def _apply_config(self, old_config: Dict[str, Any],
                      new_config: Dict[str, Any]) -> bool:
        """
        应用新配置

        Args:
            old_config: 旧配置
            new_config: 新配置

        Returns:
            bool: 是否成功应用配置
        """
        # 子类应重写此方法
        return True


class BaseResourceAwareComponent(BaseComponent,
                                 ResourceAwareComponentInterface):
    """
    基础资源感知组件 - 提供资源感知组件接口的基本实现
    实现组件的资源管理功能
    """

    def __init__(self, component_id: str, component_type: Optional[str] = None):
        """
        初始化基础资源感知组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
        """
        super().__init__(component_id, component_type)
        # 暂时不导入，先存储一个None值
        self._resource_manager = None

        # 资源分配ID列表
        self._resource_allocations = []

        # 资源需求
        self._resource_requirements = self._create_resource_requirements()

    def _get_resource_manager(self):
        """延迟获取资源管理器实例"""
        if self._resource_manager is None:
            from core.resource_manager import ResourceManager
            self._resource_manager = ResourceManager.get_instance()
        return self._resource_manager



    def get_resource_requirements(self) -> Dict[str, Any]:
        """
        获取组件资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        return dict(self._resource_requirements)

    def adapt_to_resources(self, available_resources: Dict[str, Any]) -> bool:
        """
        适应可用资源

        Args:
            available_resources: 可用资源

        Returns:
            bool: 是否成功适应
        """
        try:
            # 获取资源管理器
            resource_manager = self._get_resource_manager()

            # 获取适应建议
            adaptation_level = resource_manager.get_adaptation_level()
            suggestions = resource_manager.get_adaptation_suggestions()
            # 应用适应策略
            return self._apply_resource_adaptation(adaptation_level,
                                                   suggestions)
        except Exception as e:
            logger.error(f"资源适应错误: {self._component_id}, 错误: {e}")
            return False

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取组件资源使用情况

        Returns:
            Dict[str, Any]: 资源使用情况
        """
        # 默认实现：返回空字典
        return {}

    def _create_resource_requirements(self) -> Dict[str, Any]:
        """
        创建资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        # 子类应重写此方法
        return {}

    def _apply_resource_adaptation(self, adaptation_level,
                                   suggestions) -> bool:
        """
        应用资源适应策略

        Args:
            adaptation_level: 适应级别
            suggestions: 适应建议

        Returns:
            bool: 是否成功适应
        """
        # 子类应重写此方法
        return True

class AsyncComponentInterface(ComponentInterface):
    """
    异步组件接口 - 支持异步处理的组件接口
    定义组件的异步处理方法
    """

    @abc.abstractmethod
    def process_async(self, data: Any) -> None:
        """
        异步处理数据

        Args:
            data: 要处理的数据
        """
        pass

    @abc.abstractmethod
    def get_result(self, timeout: Optional[float] = None) -> Any:
        """
        获取处理结果

        Args:
            timeout: 超时时间（秒）

        Returns:
            Any: 处理结果
        """
        pass

    @abc.abstractmethod
    def is_busy(self) -> bool:
        """
        检查组件是否忙碌

        Returns:
            bool: 是否忙碌
        """
        pass

class BaseAsyncComponent(BaseComponent, AsyncComponentInterface):
    """
    基础异步组件 - 提供异步组件接口的基本实现
    实现组件的异步处理功能
    """

    def __init__(self, component_id: str,
                 component_type: Optional[str] = None):
        """
        初始化基础异步组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
        """
        super().__init__(component_id, component_type)

        # 异步处理队列
        self._queue = queue.Queue()

        # 结果队列
        self._result_queue = queue.Queue()

        # 工作线程
        self._worker_thread = None
        self._running = False

    def process_async(self, data: Any) -> None:
        """
        异步处理数据

        Args:
            data: 要处理的数据
        """
        try:
            # 启动工作线程（如果未启动）
            self._ensure_worker_running()

            # 添加到处理队列
            self._queue.put(data)
        except Exception as e:
            logger.error(
                f"异步处理错误: {self._component_id}, 错误: {e}")

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """
        获取处理结果

        Args:
            timeout: 超时时间（秒）

        Returns:
            Any: 处理结果
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_busy(self) -> bool:
        """
        检查组件是否忙碌

        Returns:
            bool: 是否忙碌
        """
        return not self._queue.empty()

    def _ensure_worker_running(self):
        """确保工作线程在运行"""
        # 先检查运行状态，避免重复启动
        if not self._running:
            self._running = True
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._running = True
                self._worker_thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"{self._component_id}_worker",
                    daemon=True
                )
                self._worker_thread.start()

    def _worker_loop(self):
        """工作线程循环"""
        logger.info(f"异步组件工作线程已启动: {self._component_id}")

        while self._running:
            try:
                # 获取数据（有超时以便检查running标志）
                try:
                    data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 处理数据
                try:
                    result = self._process_data(data)
                    # 添加到结果队列
                    self._result_queue.put(result)
                finally:
                    # 标记任务完成
                    self._queue.task_done()

            except Exception as e:
                logger.error(
                    f"异步处理循环错误: {self._component_id}, 错误: {e}")
                time.sleep(0.1)  # 出错时短暂等待

        logger.info(f"异步组件工作线程已停止: {self._component_id}")

    def _process_data(self, data: Any) -> Any:
        """
        处理数据

        Args:
            data: 要处理的数据

        Returns:
            Any: 处理结果
        """
        # 子类应重写此方法
        return data

    def stop(self):
        """停止异步处理"""
        self._running = False

        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        # 调用父类的 stop 方法（如果有）
        result = True
        if hasattr(super(), 'stop'):
            result = super().stop()
        return result

class FeatureToggleComponentInterface(ComponentInterface):
    """
    功能切换组件接口 - 支持功能切换的组件接口
    定义组件的功能切换方法
    """

    @abc.abstractmethod
    def toggle_feature(self, feature_name: str, enabled: bool) -> bool:
        """
        切换功能

        Args:
            feature_name: 功能名称
            enabled: 是否启用

        Returns:
            bool: 是否成功切换
        """
        pass

    @abc.abstractmethod
    def get_feature_state(self, feature_name: str) -> bool:
        """
        获取功能状态

        Args:
            feature_name: 功能名称

        Returns:
            bool: 功能是否启用
        """
        pass

    @abc.abstractmethod
    def get_available_features(self) -> Dict[str, bool]:
        """
        获取可用功能列表

        Returns:
            Dict[str, bool]: 功能名称到状态的映射
        """
        pass

class BaseFeatureToggleComponent(BaseComponent,
                                 FeatureToggleComponentInterface):
    """
    基础功能切换组件 - 提供功能切换组件接口的基本实现
    实现组件的功能切换功能
    """

    def __init__(self, component_id: str,
                 component_type: Optional[str] = None):
        """
        初始化基础功能切换组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
        """
        super().__init__(component_id, component_type)

        # 功能状态字典 {功能名称: 是否启用}
        self._features = {}

        # 初始化默认功能
        self._init_default_features()

    def toggle_feature(self, feature_name: str, enabled: bool) -> bool:
        """
        切换功能

        Args:
            feature_name: 功能名称
            enabled: 是否启用

        Returns:
            bool: 是否成功切换
        """
        try:
            # 检查功能是否存在
            if feature_name not in self._features:
                logger.warning(
                    f"功能不存在: {feature_name}, 组件: {self._component_id}")
                return False

            # 如果状态相同，不需要切换
            if self._features[feature_name] == enabled:
                logger.debug(
                    f"功能已经是目标状态: {feature_name} -> {enabled}, 组件: {self._component_id}")

                return True

            # 执行切换
            result = self._do_toggle_feature(feature_name, enabled)
            if result:
                self._features[feature_name] = enabled
                logger.debug(
                    f"功能已切换: {feature_name} -> {enabled}, 组件: {self._component_id}")

            return result
        except Exception as e:
            logger.error(
                f"切换功能错误: {feature_name}, 组件: {self._component_id}, 错误: {e}")
            return False

    def get_feature_state(self, feature_name: str) -> bool:
        """
        获取功能状态

        Args:
            feature_name: 功能名称

        Returns:
            bool: 功能是否启用
        """
        return self._features.get(feature_name, False)

    def get_available_features(self) -> Dict[str, bool]:
        """
        获取可用功能列表

        Returns:
            Dict[str, bool]: 功能名称到状态的映射
        """
        return dict(self._features)

    def _init_default_features(self):
        """初始化默认功能"""
        # 子类应重写此方法
        pass

    def _do_toggle_feature(self, feature_name: str,
                           enabled: bool) -> bool:
        """
        执行功能切换

        Args:
            feature_name: 功能名称
            enabled: 是否启用

        Returns:
            bool: 是否成功切换
        """
        # 子类应重写此方法
        return True

class CompletePipelineComponent(
    BaseLifecycleComponent,
    EventAwareComponentInterface,
    ConfigurableComponentInterface,
    FeatureToggleComponentInterface,
    ResourceAwareComponentInterface,
    AsyncComponentInterface
):
    """
    完整流水线组件 - 实现所有组件接口的基础组件
    用于构建复杂的流水线组件
    """

    def __init__(self, component_id: str,
                 component_type: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化完整流水线组件

        Args:
            component_id: 组件ID
            component_type: 组件类型，如果为None则使用类名
            config: 初始配置
        """
        # 初始化基础生命周期组件
        super().__init__(component_id, component_type)

        # 获取事件总线
        # 获取事件总线
        from core.event_bus import get_event_bus
        self._event_bus = get_event_bus()
        # 延迟导入资源管理器
        self._resource_manager = None

        # 初始化配置
        self._config = config or {}
        self._config_schema = self._create_config_schema()

        # 初始化功能
        self._features = {}
        self._init_default_features()

        # 初始化事件处理器
        self._event_handlers = {}
        self._subscriber_id = None

        # 初始化资源需求
        self._resource_requirements = self._create_resource_requirements()
        self._resource_allocations = []

        # 初始化异步处理
        self._queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._worker_thread = None
        self._running = False

    # EventAwareComponentInterface 接口实现
    def _get_resource_manager(self):
        """延迟获取资源管理器实例"""
        if self._resource_manager is None:
            from core.resource_manager import ResourceManager
            self._resource_manager = ResourceManager.get_instance()
        return self._resource_manager

    def handle_event(self, event: Event) -> bool:
        """
        处理事件

        Args:
            event: 事件对象

        Returns:
            bool: 是否成功处理事件
        """
        try:
            # 检查是否有对应的处理器
            if event.event_type in self._event_handlers:
                handler = self._event_handlers[event.event_type]
                handler(event)
                return True
            return False
        except Exception as e:
            logger.error(
                f"事件处理错误: {self._component_id}, 事件: {event.event_type.name}, 错误: {e}")
            return False

    def get_handled_event_types(self) -> Set[EventType]:
        """
        获取组件处理的事件类型

        Returns:
            Set[EventType]: 处理的事件类型集合
        """
        return set(self._event_handlers.keys())

    def subscribe_to_events(self) -> bool:
        """
        订阅事件

        Returns:
            bool: 是否成功订阅
        """
        try:
            # 取消现有订阅
            self.unsubscribe_from_events()

            # 获取处理的事件类型
            event_types = self.get_handled_event_types()
            if not event_types:
                return True  # 没有需要处理的事件类型

            # 创建事件过滤器
            from core.event_bus import EventFilter
            filter_ = EventFilter(event_types=event_types)

            # 订阅事件
            self._subscriber_id = self._event_bus.subscribe(
                event_type=None,  # 所有事件类型
                handler=self.handle_event,
                subscriber_id=f"{self._component_id}_subscriber",
                filter_=filter_
            )

            logger.debug(
                f"组件已订阅事件: {self._component_id}, 事件类型: {[t.name for t in event_types]}")
            return True
        except Exception as e:
            logger.error(
                f"事件订阅错误: {self._component_id}, 错误: {e}")
            return False

    def unsubscribe_from_events(self) -> bool:
        """
        取消订阅事件

        Returns:
            bool: 是否成功取消订阅
        """
        try:
            if self._subscriber_id:
                self._event_bus.unsubscribe(self._subscriber_id)
                self._subscriber_id = None
                logger.debug(
                    f"组件已取消订阅事件: {self._component_id}")
            return True
        except Exception as e:
            logger.error(
                f"取消事件订阅错误: {self._component_id}, 错误: {e}")
            return False

    def register_event_handler(self, event_type: EventType,
                               handler: Callable[[Event], None]):
        """
        注册事件处理器

        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        self._event_handlers[event_type] = handler

    # ConfigurableComponentInterface 接口实现

    def get_config(self) -> Dict[str, Any]:
        """
        获取组件配置

        Returns:
            Dict[str, Any]: 组件配置
        """
        return dict(self._config)

    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            config: 新配置

        Returns:
            bool: 是否成功更新配置
        """
        try:
            # 验证配置
            valid, error = self.validate_config(config)
            if not valid:
                logger.warning(
                    f"配置验证失败: {self._component_id}, 错误: {error}")
                return False

            # 更新配置
            old_config = dict(self._config)
            self._config.update(config)

            # 应用配置
            result = self._apply_config(old_config, self._config)
            if not result:
                # 恢复旧配置
                self._config = old_config
                logger.warning(f"应用配置失败: {self._component_id}")
                return False

            logger.debug(f"组件配置已更新: {self._component_id}")
            return True
        except Exception as e:
            logger.error(
                f"更新配置错误: {self._component_id}, 错误: {e}")
            return False

    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取组件配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        return dict(self._config_schema)

    def validate_config(self, config: Dict[str, Any]) -> Tuple[
        bool, Optional[str]]:
        """
        验证配置是否有效

        Args:
            config: 要验证的配置

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        try:
            # 默认实现：检查必需字段
            schema = self.get_config_schema()

            for field_name, field_info in schema.items():
                if field_info.get('required',
                                  False) and field_name not in config:
                    return False, f"缺少必需字段: {field_name}"

            return True, None
        except Exception as e:
            return False, str(e)

    # FeatureToggleComponentInterface 接口实现

    def toggle_feature(self, feature_name: str, enabled: bool) -> bool:
        """
        切换功能

        Args:
            feature_name: 功能名称
            enabled: 是否启用

        Returns:
            bool: 是否成功切换
        """
        try:
            # 检查功能是否存在
            if feature_name not in self._features:
                logger.warning(
                    f"功能不存在: {feature_name}, 组件: {self._component_id}")
                return False

            # 如果状态相同，不需要切换
            if self._features[feature_name] == enabled:
                return True

            # 执行切换
            result = self._do_toggle_feature(feature_name, enabled)
            if result:
                self._features[feature_name] = enabled
                logger.debug(
                    f"功能已切换: {feature_name} -> {enabled}, 组件: {self._component_id}")

            return result
        except Exception as e:
            logger.error(
                f"切换功能错误: {feature_name}, 组件: {self._component_id}, 错误: {e}")
            return False

    def get_feature_state(self, feature_name: str) -> bool:
        """
        获取功能状态

        Args:
            feature_name: 功能名称

        Returns:
            bool: 功能是否启用
        """
        return self._features.get(feature_name, False)

    def get_available_features(self) -> Dict[str, bool]:
        """
        获取可用功能列表

        Returns:
            Dict[str, bool]: 功能名称到状态的映射
        """
        return dict(self._features)

    # ResourceAwareComponentInterface 接口实现

    def get_resource_requirements(self) -> Dict[str, Any]:
        """
        获取组件资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        return dict(self._resource_requirements)

    def adapt_to_resources(self,
                           available_resources: Dict[str, Any]) -> bool:
        """
        适应可用资源

        Args:
            available_resources: 可用资源

        Returns:
            bool: 是否成功适应
        """
        try:
            # 获取适应建议
            adaptation_level = self._resource_manager.get_adaptation_level()
            suggestions = self._resource_manager.get_adaptation_suggestions()

            # 应用适应策略
            return self._apply_resource_adaptation(adaptation_level,
                                                   suggestions)
        except Exception as e:
            logger.error(
                f"资源适应错误: {self._component_id}, 错误: {e}")
            return False

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取组件资源使用情况

        Returns:
            Dict[str, Any]: 资源使用情况
        """
        # 默认实现：返回空字典
        return {}

    # AsyncComponentInterface 接口实现

    def process_async(self, data: Any) -> None:
        """
        异步处理数据

        Args:
            data: 要处理的数据
        """
        try:
            # 启动工作线程（如果未启动）
            self._ensure_worker_running()

            # 添加到处理队列
            self._queue.put(data)
        except Exception as e:
            logger.error(
                f"异步处理错误: {self._component_id}, 错误: {e}")

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """
        获取处理结果

        Args:
            timeout: 超时时间（秒）

        Returns:
            Any: 处理结果
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_busy(self) -> bool:
        """
        检查组件是否忙碌

        Returns:
            bool: 是否忙碌
        """
        return not self._queue.empty()

    # 辅助方法

    def _ensure_worker_running(self):
        """确保工作线程在运行"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self._component_id}_worker",
                daemon=True
            )
            self._worker_thread.start()

    def _worker_loop(self):
        """工作线程循环"""
        logger.info(f"异步组件工作线程已启动: {self._component_id}")

        while self._running:
            try:
                # 获取数据（有超时以便检查running标志）
                try:
                    data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 处理数据
                try:
                    result = self._process_data(data)
                    # 添加到结果队列
                    self._result_queue.put(result)
                finally:
                    # 标记任务完成
                    self._queue.task_done()

            except Exception as e:
                logger.error(
                    f"异步处理循环错误: {self._component_id}, 错误: {e}")
                time.sleep(0.1)  # 出错时短暂等待

        logger.info(f"异步组件工作线程已停止: {self._component_id}")

    # 子类需要实现的方法

    def _create_config_schema(self) -> Dict[str, Any]:
        """
        创建配置模式

        Returns:
            Dict[str, Any]: 配置模式
        """
        # 子类应重写此方法
        return {}

    def _apply_config(self, old_config: Dict[str, Any],
                      new_config: Dict[str, Any]) -> bool:
        """
        应用新配置

        Args:
            old_config: 旧配置
            new_config: 新配置

        Returns:
            bool: 是否成功应用配置
        """
        # 子类应重写此方法
        return True

    def _init_default_features(self):
        """初始化默认功能"""
        # 子类应重写此方法
        pass

    def _do_toggle_feature(self, feature_name: str,
                           enabled: bool) -> bool:
        """
        执行功能切换

        Args:
            feature_name: 功能名称
            enabled: 是否启用

        Returns:
            bool: 是否成功切换
        """
        # 子类应重写此方法
        return True

    def _create_resource_requirements(self) -> Dict[str, Any]:
        """
        创建资源需求

        Returns:
            Dict[str, Any]: 资源需求
        """
        # 子类应重写此方法
        return {}

    def _apply_resource_adaptation(self, adaptation_level,
                                   suggestions) -> bool:
        """
        应用资源适应策略

        Args:
            adaptation_level: 适应级别
            suggestions: 适应建议

        Returns:
            bool: 是否成功适应
        """
        # 子类应重写此方法
        return True

    def _process_data(self, data: Any) -> Any:
        """
        处理数据

        Args:
            data: 要处理的数据

        Returns:
            Any: 处理结果
        """
        # 子类应重写此方法
        return data
