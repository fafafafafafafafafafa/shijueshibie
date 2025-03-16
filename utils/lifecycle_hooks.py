# -*- coding: utf-8 -*-
"""
生命周期钩子模块 - 提供组件生命周期事件的回调机制
支持前置钩子、后置钩子、条件钩子和钩子链管理
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Set, Any, Optional, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

from core.component_lifecycle import LifecycleState

logger = logging.getLogger("LifecycleHooks")


class HookType(Enum):
    """钩子类型枚举"""
    PRE = "pre"  # 前置钩子 - 状态转换前执行
    POST = "post"  # 后置钩子 - 状态转换后执行
    CONDITION = "condition"  # 条件钩子 - 决定是否允许状态转换
    ERROR = "error"  # 错误处理钩子 - 状态转换出错时执行


@dataclass
class HookContext:
    """钩子上下文 - 包含执行钩子时的上下文信息"""
    component_id: str  # 组件ID
    component_type: str  # 组件类型
    from_state: LifecycleState  # 转换前状态
    to_state: LifecycleState  # 转换后状态
    timestamp: float = field(default_factory=time.time)  # 执行时间戳
    custom_data: Dict[str, Any] = field(default_factory=dict)  # 自定义数据
    error: Optional[Exception] = None  # 错误信息


class HookResult:
    """钩子执行结果"""

    def __init__(self, success: bool = True, message: str = "",
                 data: Any = None, should_proceed: bool = True):
        """
        初始化钩子结果

        Args:
            success: 钩子是否成功执行
            message: 结果消息
            data: 结果数据
            should_proceed: 是否应继续执行后续钩子和状态转换
        """
        self.success = success
        self.message = message
        self.data = data
        self.should_proceed = should_proceed

    @classmethod
    def success(cls, message: str = "", data: Any = None) -> 'HookResult':
        """创建成功结果"""
        return cls(True, message, data, True)

    @classmethod
    def failure(cls, message: str = "", data: Any = None) -> 'HookResult':
        """创建失败结果"""
        return cls(False, message, data, False)

    @classmethod
    def abort(cls, message: str = "", data: Any = None) -> 'HookResult':
        """创建中止结果 (成功但不继续)"""
        return cls(True, message, data, False)


@dataclass
class LifecycleHook:
    """生命周期钩子定义"""
    hook_id: str  # 钩子ID
    hook_type: HookType  # 钩子类型
    callback: Callable[[HookContext], HookResult]  # 回调函数
    states: Optional[Set[LifecycleState]] = None  # 关注的状态集合
    target_states: Optional[Set[LifecycleState]] = None  # 目标状态集合
    priority: int = 0  # 优先级，数值越高越先执行
    enabled: bool = True  # 是否启用
    description: str = ""  # 钩子描述

    def __post_init__(self):
        """初始化后处理"""
        # 如果未提供钩子ID，则自动生成
        if not self.hook_id:
            self.hook_id = f"hook_{uuid.uuid4().hex[:8]}"

        # 确保states和target_states是集合
        if self.states and not isinstance(self.states, set):
            self.states = set(self.states)
        if self.target_states and not isinstance(self.target_states, set):
            self.target_states = set(self.target_states)


class LifecycleHookManager:
    """
    生命周期钩子管理器 - 管理组件生命周期钩子
    提供钩子注册、删除、执行和链管理功能
    """

    _instance = None  # 单例实例

    @classmethod
    def get_instance(cls) -> 'LifecycleHookManager':
        """获取LifecycleHookManager单例实例"""
        if cls._instance is None:
            cls._instance = LifecycleHookManager()
        return cls._instance

    def __init__(self):
        """初始化生命周期钩子管理器"""
        if LifecycleHookManager._instance is not None:
            logger.warning(
                "LifecycleHookManager是单例类，请使用get_instance()获取实例")
            return

        # 钩子注册表 {hook_id: LifecycleHook}
        self.hooks: Dict[str, LifecycleHook] = {}

        # 组件钩子映射 {component_id: {hook_id}}
        self.component_hooks: Dict[str, Set[str]] = {}

        # 状态钩子映射 {state: {hook_id}}
        self.state_hooks: Dict[LifecycleState, Set[str]] = {}

        # 目标状态钩子映射 {target_state: {hook_id}}
        self.target_state_hooks: Dict[LifecycleState, Set[str]] = {}

        # 钩子执行历史 (最近的n条记录)
        self.max_history_size = 1000
        self.hook_history: List[Dict[str, Any]] = []

        # 线程锁
        self.lock = threading.RLock()

        # 钩子链管理
        self.hook_chains: Dict[str, List[str]] = {}

        # 条件钩子结果缓存
        self.condition_results: Dict[
            str, Dict[Tuple[LifecycleState, LifecycleState], bool]] = {}

        logger.info("生命周期钩子管理器已初始化")

    def register_hook(self, hook: Union[LifecycleHook, Dict[str, Any]],
                      component_id: Optional[str] = None) -> str:
        """
        注册生命周期钩子

        Args:
            hook: 生命周期钩子或钩子配置字典
            component_id: 关联的组件ID

        Returns:
            str: 钩子ID
        """
        with self.lock:
            # 如果是字典，转换为LifecycleHook对象
            if isinstance(hook, dict):
                # 确保有hook_type
                if 'hook_type' in hook and isinstance(hook['hook_type'], str):
                    hook['hook_type'] = HookType[hook['hook_type']]

                # 创建钩子对象
                hook = LifecycleHook(**hook)

            # 检查钩子ID是否已存在
            if hook.hook_id in self.hooks:
                logger.warning(f"钩子ID '{hook.hook_id}' 已存在，将被覆盖")
                # 如果已存在，先清除原有的映射
                self._remove_hook_mappings(hook.hook_id)

            # 添加到钩子注册表
            self.hooks[hook.hook_id] = hook

            # 更新组件映射
            if component_id:
                if component_id not in self.component_hooks:
                    self.component_hooks[component_id] = set()
                self.component_hooks[component_id].add(hook.hook_id)

            # 更新状态映射
            if hook.states:
                for state in hook.states:
                    if state not in self.state_hooks:
                        self.state_hooks[state] = set()
                    self.state_hooks[state].add(hook.hook_id)

            # 更新目标状态映射
            if hook.target_states:
                for state in hook.target_states:
                    if state not in self.target_state_hooks:
                        self.target_state_hooks[state] = set()
                    self.target_state_hooks[state].add(hook.hook_id)

            logger.debug(
                f"已注册生命周期钩子: {hook.hook_id}, 类型: {hook.hook_type.name}")
            return hook.hook_id

    def unregister_hook(self, hook_id: str) -> bool:
        """
        取消注册钩子

        Args:
            hook_id: 钩子ID

        Returns:
            bool: 是否成功取消注册
        """
        with self.lock:
            if hook_id not in self.hooks:
                logger.warning(f"尝试取消注册不存在的钩子: {hook_id}")
                return False

            # 移除钩子映射
            self._remove_hook_mappings(hook_id)

            # 从注册表中删除
            del self.hooks[hook_id]

            logger.debug(f"已取消注册生命周期钩子: {hook_id}")
            return True

    def _remove_hook_mappings(self, hook_id: str):
        """移除钩子的所有映射"""
        # 移除组件映射
        for component_id, hooks in list(self.component_hooks.items()):
            if hook_id in hooks:
                hooks.remove(hook_id)
                if not hooks:
                    del self.component_hooks[component_id]

        # 移除状态映射
        for state, hooks in list(self.state_hooks.items()):
            if hook_id in hooks:
                hooks.remove(hook_id)
                if not hooks:
                    del self.state_hooks[state]

        # 移除目标状态映射
        for state, hooks in list(self.target_state_hooks.items()):
            if hook_id in hooks:
                hooks.remove(hook_id)
                if not hooks:
                    del self.target_state_hooks[state]

        # 移除钩子链中的引用
        for chain_name, chain_hooks in list(self.hook_chains.items()):
            if hook_id in chain_hooks:
                self.hook_chains[chain_name] = [h for h in chain_hooks if
                                                h != hook_id]
                if not self.hook_chains[chain_name]:
                    del self.hook_chains[chain_name]

    def enable_hook(self, hook_id: str) -> bool:
        """
        启用钩子

        Args:
            hook_id: 钩子ID

        Returns:
            bool: 是否成功启用
        """
        with self.lock:
            if hook_id not in self.hooks:
                logger.warning(f"尝试启用不存在的钩子: {hook_id}")
                return False

            self.hooks[hook_id].enabled = True
            logger.debug(f"已启用钩子: {hook_id}")
            return True

    def disable_hook(self, hook_id: str) -> bool:
        """
        禁用钩子

        Args:
            hook_id: 钩子ID

        Returns:
            bool: 是否成功禁用
        """
        with self.lock:
            if hook_id not in self.hooks:
                logger.warning(f"尝试禁用不存在的钩子: {hook_id}")
                return False

            self.hooks[hook_id].enabled = False
            logger.debug(f"已禁用钩子: {hook_id}")
            return True

    def execute_hooks(self, hook_type: HookType, from_state: LifecycleState,
                      to_state: LifecycleState, component_id: str,
                      component_type: str, context_data: Optional[
                Dict[str, Any]] = None) -> HookResult:
        """
        执行指定类型的钩子

        Args:
            hook_type: 钩子类型
            from_state: 当前状态
            to_state: 目标状态
            component_id: 组件ID
            component_type: 组件类型
            context_data: 上下文数据

        Returns:
            HookResult: 钩子执行结果
        """
        with self.lock:
            # 创建钩子上下文
            context = HookContext(
                component_id=component_id,
                component_type=component_type,
                from_state=from_state,
                to_state=to_state,
                timestamp=time.time(),
                custom_data=context_data or {}
            )

            # 获取要执行的钩子ID集合
            hook_ids = self._get_applicable_hooks(hook_type, from_state,
                                                  to_state, component_id)

            if not hook_ids:
                # 没有可执行的钩子，返回成功
                return HookResult.success()

            # 按优先级排序钩子
            sorted_hooks = sorted(
                [self.hooks[hook_id] for hook_id in hook_ids if
                 self.hooks[hook_id].enabled],
                key=lambda h: h.priority,
                reverse=True  # 高优先级先执行
            )

            # 执行钩子并收集结果
            for hook in sorted_hooks:
                try:
                    # 执行钩子回调
                    result = hook.callback(context)

                    # 添加到历史记录
                    self._add_to_history(hook, context, result)

                    # 如果是条件钩子，缓存结果
                    if hook.hook_type == HookType.CONDITION:
                        if component_id not in self.condition_results:
                            self.condition_results[component_id] = {}
                        state_pair = (from_state, to_state)
                        self.condition_results[component_id][
                            state_pair] = result.should_proceed

                    # 如果钩子指示不应继续，立即返回
                    if not result.should_proceed:
                        logger.debug(
                            f"钩子 {hook.hook_id} 中止了执行链: {result.message}")
                        return result

                except Exception as e:
                    logger.error(f"执行钩子 {hook.hook_id} 时出错: {e}")
                    # 如果是条件钩子出错，默认不允许转换
                    if hook.hook_type == HookType.CONDITION:
                        return HookResult.failure(f"条件钩子执行出错: {e}")
                    # 其他类型钩子出错，记录但继续执行
                    context.error = e
                    self._add_to_history(hook, context,
                                         HookResult.failure(str(e)))

            # 所有钩子执行完毕，返回成功
            return HookResult.success("所有钩子执行成功")

    def _get_applicable_hooks(self, hook_type: HookType,
                              from_state: LifecycleState,
                              to_state: LifecycleState,
                              component_id: Optional[str] = None) -> Set[str]:
        """获取适用于给定状态转换的钩子ID集合"""
        # 先获取指定类型的所有钩子
        applicable_hooks = {hook_id for hook_id, hook in self.hooks.items()
                            if hook.hook_type == hook_type}

        # 如果指定了组件ID，添加该组件的钩子
        if component_id and component_id in self.component_hooks:
            applicable_hooks.update(self.component_hooks[component_id])

        # 添加与源状态相关的钩子
        if from_state in self.state_hooks:
            applicable_hooks.update(self.state_hooks[from_state])

        # 添加与目标状态相关的钩子
        if to_state in self.target_state_hooks:
            applicable_hooks.update(self.target_state_hooks[to_state])

        # 过滤钩子，只保留符合条件的
        result = set()
        for hook_id in applicable_hooks:
            if hook_id not in self.hooks:
                continue

            hook = self.hooks[hook_id]

            # 检查钩子类型
            if hook.hook_type != hook_type:
                continue

            # 检查源状态
            if hook.states and from_state not in hook.states:
                continue

            # 检查目标状态
            if hook.target_states and to_state not in hook.target_states:
                continue

            # 符合所有条件，添加到结果集
            result.add(hook_id)

        return result

    def _add_to_history(self, hook: LifecycleHook, context: HookContext,
                        result: HookResult):
        """添加钩子执行记录到历史"""
        history_entry = {
            'hook_id': hook.hook_id,
            'hook_type': hook.hook_type.name,
            'component_id': context.component_id,
            'from_state': context.from_state.name,
            'to_state': context.to_state.name,
            'timestamp': context.timestamp,
            'success': result.success,
            'message': result.message,
            'should_proceed': result.should_proceed
        }

        self.hook_history.append(history_entry)

        # 限制历史记录大小
        while len(self.hook_history) > self.max_history_size:
            self.hook_history.pop(0)

    def create_hook_chain(self, chain_name: str, hook_ids: List[str]) -> bool:
        """
        创建钩子执行链

        Args:
            chain_name: 链名称
            hook_ids: 钩子ID列表，按执行顺序排列

        Returns:
            bool: 是否成功创建
        """
        with self.lock:
            # 验证所有钩子是否存在
            for hook_id in hook_ids:
                if hook_id not in self.hooks:
                    logger.error(f"创建钩子链失败，钩子不存在: {hook_id}")
                    return False

            # 创建或更新钩子链
            self.hook_chains[chain_name] = hook_ids
            logger.info(
                f"已创建钩子链: {chain_name}, 包含 {len(hook_ids)} 个钩子")
            return True

    def execute_hook_chain(self, chain_name: str, context: HookContext) -> List[
        HookResult]:
        """
        执行钩子链

        Args:
            chain_name: 链名称
            context: 钩子上下文

        Returns:
            List[HookResult]: 钩子执行结果列表
        """
        with self.lock:
            if chain_name not in self.hook_chains:
                logger.error(f"执行钩子链失败，链不存在: {chain_name}")
                return [HookResult.failure(f"钩子链不存在: {chain_name}")]

            hook_ids = self.hook_chains[chain_name]
            results = []

            # 按顺序执行钩子
            for hook_id in hook_ids:
                if hook_id not in self.hooks:
                    logger.warning(f"钩子链中的钩子不存在: {hook_id}")
                    results.append(HookResult.failure(f"钩子不存在: {hook_id}"))
                    continue

                hook = self.hooks[hook_id]

                # 跳过禁用的钩子
                if not hook.enabled:
                    continue

                try:
                    # 执行钩子回调
                    result = hook.callback(context)
                    results.append(result)

                    # 添加到历史记录
                    self._add_to_history(hook, context, result)

                    # 如果钩子指示不应继续，中断链
                    if not result.should_proceed:
                        break

                except Exception as e:
                    logger.error(f"执行钩子链中的钩子 {hook_id} 时出错: {e}")
                    context.error = e
                    failure_result = HookResult.failure(f"钩子执行出错: {e}")
                    results.append(failure_result)
                    self._add_to_history(hook, context, failure_result)
                    break

            return results

    def get_hook_history(self, component_id: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取钩子执行历史

        Args:
            component_id: 组件ID，None表示所有组件
            limit: 最大记录数

        Returns:
            List[Dict[str, Any]]: 历史记录列表
        """
        with self.lock:
            if component_id:
                # 过滤指定组件的历史记录
                filtered_history = [
                    entry for entry in self.hook_history
                    if entry['component_id'] == component_id
                ]
                return filtered_history[-limit:]
            else:
                # 返回所有历史记录
                return self.hook_history[-limit:]

    def get_hooks_for_component(self, component_id: str) -> List[
        Dict[str, Any]]:
        """
        获取组件的所有钩子

        Args:
            component_id: 组件ID

        Returns:
            List[Dict[str, Any]]: 钩子信息列表
        """
        with self.lock:
            if component_id not in self.component_hooks:
                return []

            result = []
            for hook_id in self.component_hooks[component_id]:
                if hook_id in self.hooks:
                    hook = self.hooks[hook_id]
                    result.append({
                        'hook_id': hook.hook_id,
                        'hook_type': hook.hook_type.name,
                        'priority': hook.priority,
                        'enabled': hook.enabled,
                        'description': hook.description,
                        'states': [s.name for s in
                                   hook.states] if hook.states else None,
                        'target_states': [s.name for s in
                                          hook.target_states] if hook.target_states else None
                    })

            return result

    def create_conditional_hook(self, condition: Callable[[HookContext], bool],
                                states: Optional[Set[LifecycleState]] = None,
                                target_states: Optional[
                                    Set[LifecycleState]] = None,
                                priority: int = 0,
                                hook_id: Optional[str] = None,
                                description: str = "") -> str:
        """
        创建条件钩子

        Args:
            condition: 条件函数
            states: 关注的状态集合
            target_states: 目标状态集合
            priority: 优先级
            hook_id: 钩子ID
            description: 钩子描述

        Returns:
            str: 钩子ID
        """

        # 创建条件钩子的回调
        def condition_callback(context: HookContext) -> HookResult:
            try:
                if condition(context):
                    return HookResult.success("条件满足")
                else:
                    return HookResult.abort("条件不满足")
            except Exception as e:
                logger.error(f"条件钩子执行出错: {e}")
                return HookResult.failure(f"条件检查出错: {e}")

        # 创建钩子对象
        hook = LifecycleHook(
            hook_id=hook_id or f"condition_{uuid.uuid4().hex[:8]}",
            hook_type=HookType.CONDITION,
            callback=condition_callback,
            states=states,
            target_states=target_states,
            priority=priority,
            description=description or "条件钩子"
        )

        # 注册钩子
        return self.register_hook(hook)

    def create_logging_hook(self, log_level: int = logging.INFO,
                            states: Optional[Set[LifecycleState]] = None,
                            target_states: Optional[Set[LifecycleState]] = None,
                            priority: int = 0,
                            hook_id: Optional[str] = None,
                            hook_type: HookType = HookType.POST,
                            description: str = "") -> str:
        """
        创建日志钩子

        Args:
            log_level: 日志级别
            states: 关注的状态集合
            target_states: 目标状态集合
            priority: 优先级
            hook_id: 钩子ID
            hook_type: 钩子类型
            description: 钩子描述

        Returns:
            str: 钩子ID
        """

        # 创建日志钩子的回调
        def logging_callback(context: HookContext) -> HookResult:
            log_msg = (
                f"组件状态转换: {context.component_type} ({context.component_id}) "
                f"{context.from_state.name} -> {context.to_state.name}")
            logger.log(log_level, log_msg)
            return HookResult.success()

        # 创建钩子对象
        hook = LifecycleHook(
            hook_id=hook_id or f"logging_{uuid.uuid4().hex[:8]}",
            hook_type=hook_type,
            callback=logging_callback,
            states=states,
            target_states=target_states,
            priority=priority,
            description=description or "日志钩子"
        )

        # 注册钩子
        return self.register_hook(hook)

    def create_cleanup_hook(self, cleanup_func: Callable[[HookContext], None],
                            states: Optional[Set[LifecycleState]] = None,
                            target_states: Optional[Set[LifecycleState]] = None,
                            priority: int = 0,
                            hook_id: Optional[str] = None,
                            description: str = "") -> str:
        """
        创建清理钩子

        Args:
            cleanup_func: 清理函数
            states: 关注的状态集合
            target_states: 目标状态集合
            priority: 优先级
            hook_id: 钩子ID
            description: 钩子描述

        Returns:
            str: 钩子ID
        """

        # 创建清理钩子的回调
        def cleanup_callback(context: HookContext) -> HookResult:
            try:
                cleanup_func(context)
                return HookResult.success("清理完成")
            except Exception as e:
                logger.error(f"清理钩子执行出错: {e}")
                return HookResult.failure(f"清理出错: {e}")

        # 创建钩子对象
        hook = LifecycleHook(
            hook_id=hook_id or f"cleanup_{uuid.uuid4().hex[:8]}",
            hook_type=HookType.POST,
            callback=cleanup_callback,
            states=states,
            target_states=target_states,
            priority=priority,
            description=description or "清理钩子"
        )

        # 注册钩子
        return self.register_hook(hook)

    def clear(self):
        """清除所有钩子"""
        with self.lock:
            self.hooks.clear()
            self.component_hooks.clear()
            self.state_hooks.clear()
            self.target_state_hooks.clear()
            self.hook_chains.clear()
            self.hook_history.clear()
            self.condition_results.clear()
            logger.info("已清除所有钩子")


# 便捷函数 - 获取生命周期钩子管理器单例
def get_hook_manager() -> LifecycleHookManager:
    """获取生命周期钩子管理器单例实例"""
    return LifecycleHookManager.get_instance()
