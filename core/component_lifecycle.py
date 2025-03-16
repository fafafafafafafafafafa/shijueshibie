# -*- coding: utf-8 -*-
"""
组件生命周期管理模块 - 定义并管理组件生命周期状态
提供统一的生命周期接口和事件处理机制
"""

import time
import logging
import threading
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from .event_models import Event, EventType, EventMetadata, EventPriority
from .event_bus import get_event_bus, EventBus

# 设置日志
logger = logging.getLogger("ComponentLifecycle")


class LifecycleState(Enum):
    """组件生命周期状态枚举"""
    UNREGISTERED = auto()  # 未注册 - 组件存在但未向系统注册
    REGISTERED = auto()  # 已注册 - 组件已注册但未初始化
    INITIALIZING = auto()  # 初始化中 - 组件正在初始化
    INITIALIZED = auto()  # 已初始化 - 组件已初始化但未启动
    STARTING = auto()  # 启动中 - 组件正在启动
    RUNNING = auto()  # 运行中 - 组件正在运行
    PAUSING = auto()  # 暂停中 - 组件正在暂停
    PAUSED = auto()  # 已暂停 - 组件已暂停
    STOPPING = auto()  # 停止中 - 组件正在停止
    STOPPED = auto()  # 已停止 - 组件已停止
    DESTROYING = auto()  # 销毁中 - 组件正在销毁
    DESTROYED = auto()  # 已销毁 - 组件已销毁
    ERROR = auto()  # 错误状态 - 组件处于错误状态


class LifecycleTransition:
    """生命周期状态转换"""
    # 定义有效的状态转换
    VALID_TRANSITIONS = {
        LifecycleState.UNREGISTERED: {LifecycleState.REGISTERED},
        LifecycleState.REGISTERED: {LifecycleState.INITIALIZING,
                                    LifecycleState.ERROR},
        LifecycleState.INITIALIZING: {LifecycleState.INITIALIZED,
                                      LifecycleState.ERROR},
        LifecycleState.INITIALIZED: {LifecycleState.STARTING,
                                     LifecycleState.DESTROYING,
                                     LifecycleState.ERROR},
        LifecycleState.STARTING: {LifecycleState.RUNNING, LifecycleState.ERROR},
        LifecycleState.RUNNING: {LifecycleState.PAUSING,
                                 LifecycleState.STOPPING, LifecycleState.ERROR},
        LifecycleState.PAUSING: {LifecycleState.PAUSED, LifecycleState.ERROR},
        LifecycleState.PAUSED: {LifecycleState.STARTING,
                                LifecycleState.STOPPING, LifecycleState.ERROR},
        LifecycleState.STOPPING: {LifecycleState.STOPPED, LifecycleState.ERROR},
        LifecycleState.STOPPED: {LifecycleState.STARTING,
                                 LifecycleState.DESTROYING,
                                 LifecycleState.ERROR},
        LifecycleState.DESTROYING: {LifecycleState.DESTROYED,
                                    LifecycleState.ERROR},
        LifecycleState.DESTROYED: {LifecycleState.UNREGISTERED},
        LifecycleState.ERROR: {LifecycleState.STOPPING,
                               LifecycleState.DESTROYING,
                               LifecycleState.INITIALIZED}
    }

    @classmethod
    def is_valid_transition(cls, from_state: LifecycleState,
                            to_state: LifecycleState) -> bool:
        """
        检查状态转换是否有效

        Args:
            from_state: 当前状态
            to_state: 目标状态

        Returns:
            bool: 状态转换是否有效
        """
        # 如果状态相同，视为有效转换（不变）
        if from_state == to_state:
            return True

        # 检查是否在有效转换列表中
        return to_state in cls.VALID_TRANSITIONS.get(from_state, set())

    @classmethod
    def get_valid_next_states(cls, current_state: LifecycleState) -> Set[
        LifecycleState]:
        """
        获取当前状态的所有有效后续状态

        Args:
            current_state: 当前状态

        Returns:
            Set[LifecycleState]: 有效的后续状态集合
        """
        return cls.VALID_TRANSITIONS.get(current_state, set())

    @classmethod
    def get_valid_transitions(cls) -> Dict[LifecycleState, Set[LifecycleState]]:
        """
        获取所有有效状态转换

        Returns:
            Dict[LifecycleState, Set[LifecycleState]]: 状态转换映射
        """
        return cls.VALID_TRANSITIONS.copy()


@dataclass
class LifecycleHook:
    """生命周期钩子 - 在状态转换前后执行"""
    hook_id: str
    callback: Callable[[LifecycleState, LifecycleState, Optional[Any]], None]
    pre_transition: bool = False  # True表示在转换前执行，False表示在转换后执行
    states: Optional[Set[LifecycleState]] = None  # 关注的状态，None表示所有状态
    priority: int = 0  # 优先级，数值越高越先执行

    def __post_init__(self):
        """初始化后的处理"""
        if self.states is not None and not isinstance(self.states, set):
            if isinstance(self.states, (list, tuple)):
                self.states = set(self.states)
            else:
                self.states = {self.states}

    def should_execute(self, from_state: LifecycleState,
                       to_state: LifecycleState) -> bool:
        """
        检查钩子是否应该被执行

        Args:
            from_state: 转换前状态
            to_state: 转换后状态

        Returns:
            bool: 是否应该执行
        """
        # 如果未指定关注状态，则对所有状态执行
        if self.states is None:
            return True

        # 根据钩子类型检查是否应该执行
        if self.pre_transition:
            # 前置钩子关注转换前状态
            return from_state in self.states
        else:
            # 后置钩子关注转换后状态
            return to_state in self.states


class LifecycleManager:
    """
    生命周期管理器 - 管理组件的生命周期状态
    提供状态转换、钩子注册和状态查询功能
    """

    def __init__(self, component_id: str, component_name: str,
                 initial_state: LifecycleState = LifecycleState.UNREGISTERED):
        """
        初始化生命周期管理器

        Args:
            component_id: 组件ID
            component_name: 组件名称
            initial_state: 初始状态
        """
        self.component_id = component_id
        self.component_name = component_name
        self.current_state = initial_state
        self.state_history: List[Tuple[LifecycleState, float]] = [
            (initial_state, time.time())]
        self.hooks: Dict[str, LifecycleHook] = {}
        self.lock = threading.RLock()

        # 获取事件总线
        self.event_bus = get_event_bus()

        # 状态转换时间统计
        self.state_durations: Dict[LifecycleState, float] = {}
        self.last_transition_time = time.time()

        # 注册到系统（如果不是未注册状态）
        if initial_state != LifecycleState.UNREGISTERED:
            self._publish_state_change_event(LifecycleState.UNREGISTERED,
                                             initial_state)

        logger.info(
            f"生命周期管理器初始化: {component_id} - {component_name}, 初始状态: {initial_state.name}")

    def get_current_state(self) -> LifecycleState:
        """
        获取当前状态

        Returns:
            LifecycleState: 当前生命周期状态
        """
        with self.lock:
            return self.current_state

    def get_state_history(self) -> List[Tuple[LifecycleState, float]]:
        """
        获取状态历史

        Returns:
            List[Tuple[LifecycleState, float]]: 状态历史列表 [(状态, 时间戳), ...]
        """
        with self.lock:
            return list(self.state_history)

    def get_current_state_duration(self) -> float:
        """
        获取当前状态持续时间（秒）

        Returns:
            float: 当前状态持续时间
        """
        with self.lock:
            return time.time() - self.last_transition_time

    def get_total_uptime(self) -> float:
        """
        获取组件总运行时间（处于RUNNING状态的时间总和）

        Returns:
            float: 总运行时间（秒）
        """
        with self.lock:
            total_time = self.state_durations.get(LifecycleState.RUNNING, 0)
            # 如果当前状态是RUNNING，添加当前会话的运行时间
            if self.current_state == LifecycleState.RUNNING:
                total_time += time.time() - self.last_transition_time
            return total_time

    def get_state_durations(self) -> Dict[LifecycleState, float]:
        """
        获取各状态持续时间统计

        Returns:
            Dict[LifecycleState, float]: 各状态持续时间字典
        """
        with self.lock:
            # 复制状态时间字典
            durations = self.state_durations.copy()

            # 添加当前状态的持续时间
            current_duration = time.time() - self.last_transition_time
            if self.current_state in durations:
                durations[self.current_state] += current_duration
            else:
                durations[self.current_state] = current_duration

            return durations

    def register_hook(self, hook: LifecycleHook) -> bool:
        """
        注册生命周期钩子

        Args:
            hook: 要注册的钩子

        Returns:
            bool: 是否成功注册
        """
        with self.lock:
            if hook.hook_id in self.hooks:
                logger.warning(f"钩子ID '{hook.hook_id}' 已存在，无法注册")
                return False

            self.hooks[hook.hook_id] = hook
            logger.debug(
                f"已注册生命周期钩子: {hook.hook_id}, 优先级: {hook.priority}, "
                f"阶段: {'前置' if hook.pre_transition else '后置'}")
            return True

    def unregister_hook(self, hook_id: str) -> bool:
        """
        取消注册生命周期钩子

        Args:
            hook_id: 钩子ID

        Returns:
            bool: 是否成功取消注册
        """
        with self.lock:
            if hook_id not in self.hooks:
                logger.warning(f"钩子ID '{hook_id}' 不存在，无法取消注册")
                return False

            del self.hooks[hook_id]
            logger.debug(f"已取消注册生命周期钩子: {hook_id}")
            return True

    def transition_to(self, target_state: LifecycleState,
                      context: Optional[Any] = None) -> bool:
        """
        将组件转换到新状态

        Args:
            target_state: 目标状态
            context: 转换上下文信息

        Returns:
            bool: 是否成功转换
        """
        with self.lock:
            current_state = self.current_state

            # 检查转换是否有效
            if not LifecycleTransition.is_valid_transition(current_state,
                                                           target_state):
                logger.warning(
                    f"无效的状态转换: {current_state.name} -> {target_state.name}, "
                    f"组件: {self.component_name} ({self.component_id})"
                )
                return False

            # 如果状态相同，则不进行转换
            if current_state == target_state:
                logger.debug(
                    f"状态未变，跳过转换: {current_state.name}, 组件: {self.component_name}")
                return True

            # 更新状态持续时间统计
            now = time.time()
            duration = now - self.last_transition_time
            if current_state in self.state_durations:
                self.state_durations[current_state] += duration
            else:
                self.state_durations[current_state] = duration

            # 排序前置钩子（按优先级降序）
            pre_hooks = [
                hook for hook in self.hooks.values()
                if hook.pre_transition and hook.should_execute(current_state,
                                                               target_state)
            ]
            pre_hooks.sort(key=lambda h: h.priority, reverse=True)

            # 执行前置钩子
            for hook in pre_hooks:
                try:
                    hook.callback(current_state, target_state, context)
                except Exception as e:
                    logger.error(
                        f"执行前置钩子时出错: {hook.hook_id}, 错误: {e}")

            # 执行状态转换
            self.current_state = target_state
            self.state_history.append((target_state, now))
            self.last_transition_time = now

            # 排序后置钩子（按优先级降序）
            post_hooks = [
                hook for hook in self.hooks.values()
                if
                not hook.pre_transition and hook.should_execute(current_state,
                                                                target_state)
            ]
            post_hooks.sort(key=lambda h: h.priority, reverse=True)

            # 执行后置钩子
            for hook in post_hooks:
                try:
                    hook.callback(current_state, target_state, context)
                except Exception as e:
                    logger.error(
                        f"执行后置钩子时出错: {hook.hook_id}, 错误: {e}")

            # 发布状态变更事件
            self._publish_state_change_event(current_state, target_state,
                                             context)

            logger.info(
                f"组件状态已转换: {current_state.name} -> {target_state.name}, "
                f"组件: {self.component_name} ({self.component_id})")

            return True

    def _publish_state_change_event(self, from_state: LifecycleState,
                                    to_state: LifecycleState,
                                    context: Any = None):
        """
        发布状态变更事件

        Args:
            from_state: 转换前状态
            to_state: 转换后状态
            context: 上下文信息
        """
        # 准备事件数据
        event_data = {
            'component_id': self.component_id,
            'component_name': self.component_name,
            'from_state': from_state.name,
            'to_state': to_state.name,
            'timestamp': time.time(),
            'context': context
        }

        # 发布状态变更事件
        self.event_bus.create_and_publish(
            event_type=EventType.COMPONENT_STATE_CHANGED,
            data=event_data,
            source_id=self.component_id,
            priority=EventPriority.NORMAL
        )

        # 发布特定状态事件
        event_type_map = {
            LifecycleState.INITIALIZED: EventType.COMPONENT_INITIALIZED,
            LifecycleState.RUNNING: EventType.COMPONENT_STARTED,
            LifecycleState.STOPPED: EventType.COMPONENT_STOPPED,
            LifecycleState.ERROR: EventType.COMPONENT_ERROR
        }

        if to_state in event_type_map:
            self.event_bus.create_and_publish(
                event_type=event_type_map[to_state],
                data=event_data,
                source_id=self.component_id,
                priority=EventPriority.NORMAL
            )

    def can_transition_to(self, target_state: LifecycleState) -> bool:
        """
        检查是否可以转换到目标状态

        Args:
            target_state: 目标状态

        Returns:
            bool: 是否可以转换
        """
        with self.lock:
            return LifecycleTransition.is_valid_transition(self.current_state,
                                                           target_state)

    def get_next_states(self) -> Set[LifecycleState]:
        """
        获取当前状态的所有有效后续状态

        Returns:
            Set[LifecycleState]: 有效的后续状态集合
        """
        with self.lock:
            return LifecycleTransition.get_valid_next_states(self.current_state)

    def set_error(self, error_info: Optional[Any] = None) -> bool:
        """
        将组件设置为错误状态

        Args:
            error_info: 错误信息

        Returns:
            bool: 是否成功设置
        """
        with self.lock:
            # 如果当前已经是错误状态，只更新错误信息
            if self.current_state == LifecycleState.ERROR:
                # 发布错误事件
                self.event_bus.create_and_publish(
                    event_type=EventType.COMPONENT_ERROR,
                    data={
                        'component_id': self.component_id,
                        'component_name': self.component_name,
                        'error_info': error_info,
                        'timestamp': time.time()
                    },
                    source_id=self.component_id,
                    priority=EventPriority.HIGH
                )
                return True

            # 尝试转换到错误状态
            return self.transition_to(LifecycleState.ERROR, error_info)

    def create_standard_lifecycle(self) -> Dict[str, Callable[[], bool]]:
        """
        创建标准生命周期方法字典

        Returns:
            Dict[str, Callable]: 标准生命周期方法字典
        """
        return {
            'initialize': lambda: self.transition_to(
                LifecycleState.INITIALIZED),
            'start': lambda: self.transition_to(LifecycleState.RUNNING),
            'pause': lambda: self.transition_to(LifecycleState.PAUSED),
            'resume': lambda: self.transition_to(LifecycleState.RUNNING),
            'stop': lambda: self.transition_to(LifecycleState.STOPPED),
            'destroy': lambda: self.transition_to(LifecycleState.DESTROYED)
        }


class LifecycleAware:
    """
    生命周期感知接口 - 组件实现此接口以获得标准生命周期管理
    提供组件生命周期管理的基础功能
    """

    def __init__(self, component_id: str, component_name: str):
        """
        初始化生命周期感知组件

        Args:
            component_id: 组件ID
            component_name: 组件名称
        """
        self.component_id = component_id
        self.component_name = component_name

        # 创建生命周期管理器
        self.lifecycle_manager = LifecycleManager(component_id, component_name)

        # 获取事件总线
        self.event_bus = get_event_bus()

        # 注册到系统
        self.lifecycle_manager.transition_to(LifecycleState.REGISTERED)

    def initialize(self) -> bool:
        """
        初始化组件

        Returns:
            bool: 是否成功初始化
        """
        try:
            # 转换到初始化中状态
            self.lifecycle_manager.transition_to(LifecycleState.INITIALIZING)

            # 调用实际初始化逻辑
            result = self._do_initialize()

            # 根据结果转换状态
            if result:
                self.lifecycle_manager.transition_to(LifecycleState.INITIALIZED)
                return True
            else:
                self.lifecycle_manager.set_error("初始化失败")
                return False
        except Exception as e:
            logger.error(
                f"组件初始化时出错: {self.component_name} ({self.component_id}), 错误: {e}")
            self.lifecycle_manager.set_error(str(e))
            return False

    def start(self) -> bool:
        """
        启动组件

        Returns:
            bool: 是否成功启动
        """
        try:
            # 检查当前状态是否允许启动
            current_state = self.lifecycle_manager.get_current_state()
            if current_state not in [LifecycleState.INITIALIZED,
                                     LifecycleState.STOPPED,
                                     LifecycleState.PAUSED]:
                logger.warning(
                    f"组件无法从 {current_state.name} 状态启动: {self.component_name}")
                return False

            # 转换到启动中状态
            self.lifecycle_manager.transition_to(LifecycleState.STARTING)

            # 调用实际启动逻辑
            result = self._do_start()

            # 根据结果转换状态
            if result:
                self.lifecycle_manager.transition_to(LifecycleState.RUNNING)
                return True
            else:
                self.lifecycle_manager.set_error("启动失败")
                return False
        except Exception as e:
            logger.error(
                f"组件启动时出错: {self.component_name} ({self.component_id}), 错误: {e}")
            self.lifecycle_manager.set_error(str(e))
            return False

    def pause(self) -> bool:
        """
        暂停组件

        Returns:
            bool: 是否成功暂停
        """
        try:
            # 检查当前状态是否允许暂停
            if self.lifecycle_manager.get_current_state() != LifecycleState.RUNNING:
                logger.warning(
                    f"组件未运行，无法暂停: {self.component_name}")
                return False

            # 转换到暂停中状态
            self.lifecycle_manager.transition_to(LifecycleState.PAUSING)

            # 调用实际暂停逻辑
            result = self._do_pause()

            # 根据结果转换状态
            if result:
                self.lifecycle_manager.transition_to(LifecycleState.PAUSED)
                return True
            else:
                self.lifecycle_manager.set_error("暂停失败")
                return False
        except Exception as e:
            logger.error(
                f"组件暂停时出错: {self.component_name} ({self.component_id}), 错误: {e}")
            self.lifecycle_manager.set_error(str(e))
            return False

    def resume(self) -> bool:
        """
        恢复组件

        Returns:
            bool: 是否成功恢复
        """
        # 如果组件已暂停，则启动它
        if self.lifecycle_manager.get_current_state() == LifecycleState.PAUSED:
            return self.start()

        logger.warning(f"组件未暂停，无法恢复: {self.component_name}")
        return False

    def stop(self) -> bool:
        """
        停止组件

        Returns:
            bool: 是否成功停止
        """
        try:
            # 检查当前状态是否允许停止
            current_state = self.lifecycle_manager.get_current_state()
            if current_state not in [LifecycleState.RUNNING,
                                     LifecycleState.PAUSED,
                                     LifecycleState.ERROR]:
                logger.warning(
                    f"组件无法从 {current_state.name} 状态停止: {self.component_name}")
                return False

            # 转换到停止中状态
            self.lifecycle_manager.transition_to(LifecycleState.STOPPING)

            # 调用实际停止逻辑
            result = self._do_stop()

            # 根据结果转换状态
            if result:
                self.lifecycle_manager.transition_to(LifecycleState.STOPPED)
                return True
            else:
                self.lifecycle_manager.set_error("停止失败")
                return False
        except Exception as e:
            logger.error(
                f"组件停止时出错: {self.component_name} ({self.component_id}), 错误: {e}")
            self.lifecycle_manager.set_error(str(e))
            return False

    def destroy(self) -> bool:
        """
        销毁组件

        Returns:
            bool: 是否成功销毁
        """
        try:
            # 检查当前状态是否允许销毁
            current_state = self.lifecycle_manager.get_current_state()
            if current_state not in [LifecycleState.INITIALIZED,
                                     LifecycleState.STOPPED,
                                     LifecycleState.ERROR]:
                logger.warning(
                    f"组件无法从 {current_state.name} 状态销毁: {self.component_name}")
                return False

            # 转换到销毁中状态
            self.lifecycle_manager.transition_to(LifecycleState.DESTROYING)

            # 调用实际销毁逻辑
            result = self._do_destroy()

            # 根据结果转换状态
            if result:
                self.lifecycle_manager.transition_to(
                    LifecycleState.DESTROYED)
                return True
            else:
                self.lifecycle_manager.set_error("销毁失败")
                return False
        except Exception as e:
            logger.error(
                f"组件销毁时出错: {self.component_name} ({self.component_id}), 错误: {e}")
            self.lifecycle_manager.set_error(str(e))
            return False

    def get_state(self) -> LifecycleState:
        """
        获取组件当前状态

        Returns:
            LifecycleState: 当前生命周期状态
        """
        return self.lifecycle_manager.get_current_state()

    def is_running(self) -> bool:
        """
        检查组件是否运行中

        Returns:
            bool: 是否运行中
        """
        return self.lifecycle_manager.get_current_state() == LifecycleState.RUNNING

    def is_error(self) -> bool:
        """
        检查组件是否处于错误状态

        Returns:
            bool: 是否错误状态
        """
        return self.lifecycle_manager.get_current_state() == LifecycleState.ERROR

    # 以下方法由子类实现

    def _do_initialize(self) -> bool:
        """
        执行实际的初始化逻辑
        子类必须重写此方法

        Returns:
            bool: 是否成功初始化
        """
        raise NotImplementedError("子类必须实现_do_initialize方法")

    def _do_start(self) -> bool:
        """
        执行实际的启动逻辑
        子类必须重写此方法

        Returns:
            bool: 是否成功启动
        """
        raise NotImplementedError("子类必须实现_do_start方法")

    def _do_pause(self) -> bool:
        """
        执行实际的暂停逻辑
        子类必须重写此方法

        Returns:
            bool: 是否成功暂停
        """
        raise NotImplementedError("子类必须实现_do_pause方法")

    def _do_stop(self) -> bool:
        """
        执行实际的停止逻辑
        子类必须重写此方法

        Returns:
            bool: 是否成功停止
        """
        raise NotImplementedError("子类必须实现_do_stop方法")

    def _do_destroy(self) -> bool:
        """
        执行实际的销毁逻辑
        子类必须重写此方法

        Returns:
            bool: 是否成功销毁
        """
        raise NotImplementedError("子类必须实现_do_destroy方法")

class AbstractLifecycleComponent(LifecycleAware):
    """
    抽象生命周期组件 - 提供生命周期方法的基本实现
    子类可以选择性地重写需要的方法
    """

    def _do_initialize(self) -> bool:
        """基本初始化实现"""
        logger.debug(f"组件初始化（默认实现）: {self.component_name}")
        logger.warning(
            f"{self.__class__.__name__} 未实现 _do_initialize 方法")
        return True

    def _do_start(self) -> bool:
        """基本启动实现"""
        logger.debug(f"组件启动（默认实现）: {self.component_name}")
        return True

    def _do_pause(self) -> bool:
        """基本暂停实现"""
        logger.debug(f"组件暂停（默认实现）: {self.component_name}")
        return True

    def _do_stop(self) -> bool:
        """基本停止实现"""
        logger.debug(f"组件停止（默认实现）: {self.component_name}")
        return True

    def _do_destroy(self) -> bool:
        """基本销毁实现"""
        logger.debug(f"组件销毁（默认实现）: {self.component_name}")
        return True

def create_lifecycle_manager(component_id: str,
                             component_name: str) -> LifecycleManager:
    """
    创建生命周期管理器的工厂函数

    Args:
        component_id: 组件ID
        component_name: 组件名称

    Returns:
        LifecycleManager: 新的生命周期管理器实例
    """
    return LifecycleManager(component_id, component_name)

def create_lifecycle_hook(
        hook_id: str,
        callback: Callable[
            [LifecycleState, LifecycleState, Optional[Any]], None],
        pre_transition: bool = False,
        states: Optional[Set[LifecycleState]] = None,
        priority: int = 0
) -> LifecycleHook:
    """
    创建生命周期钩子的工厂函数

    Args:
        hook_id: 钩子ID
        callback: 回调函数
        pre_transition: 是否前置钩子
        states: 关注的状态集合
        priority: 优先级

    Returns:
        LifecycleHook: 新的生命周期钩子实例
    """
    return LifecycleHook(
        hook_id=hook_id,
        callback=callback,
        pre_transition=pre_transition,
        states=states,
        priority=priority
    )
