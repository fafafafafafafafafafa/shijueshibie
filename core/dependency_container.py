# -*- coding: utf-8 -*-
"""
依赖注入容器模块 - 提供组件注册、查找和依赖注入功能
支持单例、工厂和按需创建模式
"""

import inspect
import logging
import threading
from enum import Enum, auto
from typing import Dict, List, Set, Any, Optional, Callable, Type, TypeVar, \
    cast, get_type_hints
from dataclasses import dataclass, field
#from .dependency_container import get_container, DependencyContainer
import weakref

# 设置日志
logger = logging.getLogger("DependencyContainer")

# 类型变量，用于泛型函数
T = TypeVar('T')


class InjectionScope(Enum):
    """依赖注入作用域枚举"""
    SINGLETON = auto()  # 单例 - 整个应用程序中只有一个实例
    TRANSIENT = auto()  # 瞬态 - 每次请求都创建新实例
    SCOPED = auto()  # 作用域 - 在同一作用域内是单例


@dataclass
class DependencyRegistration:
    """依赖注册信息"""
    interface_type: Type  # 接口类型
    implementation_type: Optional[Type] = None  # 实现类型
    instance: Any = None  # 单例实例
    factory: Optional[Callable[..., Any]] = None  # 工厂函数
    scope: InjectionScope = InjectionScope.SINGLETON  # 注入作用域
    name: Optional[str] = None  # 命名依赖
    tags: Set[str] = field(default_factory=set)  # 标签
    dependencies: Dict[str, Any] = field(default_factory=dict)  # 显式依赖


class DependencyContainer:
    """
    依赖注入容器 - 管理组件依赖和生命周期
    提供依赖注册、解析和自动注入功能
    """

    _instance = None  # 单例实例

    @classmethod
    def get_instance(cls) -> 'DependencyContainer':
        """获取DependencyContainer单例实例"""
        if cls._instance is None:
            cls._instance = DependencyContainer()
        return cls._instance

    def __init__(self):
        """初始化依赖注入容器"""
        if DependencyContainer._instance is not None:
            logger.warning(
                "DependencyContainer是单例类，请使用get_instance()获取实例")
            return

        # 注册表 {接口类型: {名称: 注册信息}}
        self._registrations: Dict[
            Type, Dict[Optional[str], DependencyRegistration]] = {}

        # 当前解析栈，用于检测循环依赖
        self._resolution_stack: List[str] = []

        # 作用域字典 {作用域ID: {接口类型: {名称: 实例}}}
        self._scoped_instances: Dict[
            str, Dict[Type, Dict[Optional[str], Any]]] = {}

        # 当前作用域ID
        self._current_scope_id: Optional[str] = None

        # 线程锁
        self._lock = threading.RLock()

        logger.info("依赖注入容器已初始化")

    def register_instance(self,
                          interface_type: Type[T],
                          instance: T,
                          name: Optional[str] = None,
                          tags: Optional[Set[str]] = None) -> None:
        """
        注册已存在的实例

        Args:
            interface_type: 接口类型
            instance: 实例对象
            name: 可选的命名依赖
            tags: 可选的标签集合
        """
        with self._lock:
            # 确保接口类型在注册表中
            if interface_type not in self._registrations:
                self._registrations[interface_type] = {}

            # 创建注册信息
            registration = DependencyRegistration(
                interface_type=interface_type,
                instance=instance,
                scope=InjectionScope.SINGLETON,  # 实例总是单例
                name=name,
                tags=tags or set()
            )

            # 添加到注册表
            self._registrations[interface_type][name] = registration

            logger.debug(
                f"已注册实例: {interface_type.__name__}{f' ({name})' if name else ''}")

    def register_type(self,
                      interface_type: Type[T],
                      implementation_type: Type[T],
                      scope: InjectionScope = InjectionScope.SINGLETON,
                      name: Optional[str] = None,
                      tags: Optional[Set[str]] = None,
                      dependencies: Optional[Dict[str, Any]] = None) -> None:
        """
        注册类型映射

        Args:
            interface_type: 接口类型
            implementation_type: 实现类型
            scope: 注入作用域
            name: 可选的命名依赖
            tags: 可选的标签集合
            dependencies: 显式依赖字典
        """
        with self._lock:
            # 确保接口类型在注册表中
            if interface_type not in self._registrations:
                self._registrations[interface_type] = {}

            # 创建注册信息
            registration = DependencyRegistration(
                interface_type=interface_type,
                implementation_type=implementation_type,
                scope=scope,
                name=name,
                tags=tags or set(),
                dependencies=dependencies or {}
            )

            # 如果是单例且未提供依赖，则立即创建实例
            if scope == InjectionScope.SINGLETON and not dependencies:
                try:
                    instance = self._create_instance(implementation_type)
                    registration.instance = instance
                except Exception as e:
                    logger.error(
                        f"创建单例实例时出错: {implementation_type.__name__}, 错误: {e}")

            # 添加到注册表
            self._registrations[interface_type][name] = registration

            logger.debug(
                f"已注册类型: {interface_type.__name__} -> {implementation_type.__name__}"
                f"{f' ({name})' if name else ''}, 作用域: {scope.name}")

    def register_factory(self,
                         interface_type: Type[T],
                         factory: Callable[..., T],
                         scope: InjectionScope = InjectionScope.SINGLETON,
                         name: Optional[str] = None,
                         tags: Optional[Set[str]] = None,
                         dependencies: Optional[Dict[str, Any]] = None) -> None:
        """
        注册工厂函数

        Args:
            interface_type: 接口类型
            factory: 工厂函数
            scope: 注入作用域
            name: 可选的命名依赖
            tags: 可选的标签集合
            dependencies: 显式依赖字典
        """
        with self._lock:
            # 确保接口类型在注册表中
            if interface_type not in self._registrations:
                self._registrations[interface_type] = {}

            # 创建注册信息
            registration = DependencyRegistration(
                interface_type=interface_type,
                factory=factory,
                scope=scope,
                name=name,
                tags=tags or set(),
                dependencies=dependencies or {}
            )

            # 如果是单例且未提供依赖，则立即创建实例
            if scope == InjectionScope.SINGLETON and not dependencies:
                try:
                    instance = factory()
                    registration.instance = instance
                except Exception as e:
                    logger.error(
                        f"创建单例实例时出错: {interface_type.__name__}, 错误: {e}")

            # 添加到注册表
            self._registrations[interface_type][name] = registration

            logger.debug(
                f"已注册工厂: {interface_type.__name__}{f' ({name})' if name else ''}, "
                f"作用域: {scope.name}")

    def register_self(self) -> None:
        """注册容器自身，使其可被注入"""
        self.register_instance(DependencyContainer, self)

    def resolve(self, interface_type: Type[T], name: Optional[str] = None) -> T:
        """
        解析依赖

        Args:
            interface_type: 接口类型
            name: 可选的命名依赖

        Returns:
            T: 解析的实例

        Raises:
            KeyError: 找不到注册的依赖
            ValueError: 解析过程中出错
        """
        with self._lock:
            # 构造解析标识
            resolution_id = f"{interface_type.__name__}{f'({name})' if name else ''}"

            # 检测循环依赖
            if resolution_id in self._resolution_stack:
                cycle = " -> ".join(
                    self._resolution_stack) + f" -> {resolution_id}"
                raise ValueError(f"检测到循环依赖: {cycle}")

            # 加入解析栈
            self._resolution_stack.append(resolution_id)

            try:
                # 查找注册信息
                if interface_type not in self._registrations:
                    raise KeyError(f"未找到类型注册: {interface_type.__name__}")

                if name not in self._registrations[interface_type]:
                    if None not in self._registrations[interface_type]:
                        raise KeyError(
                            f"未找到命名依赖: {interface_type.__name__}({name})")
                    # 使用默认注册
                    name = None

                registration = self._registrations[interface_type][name]

                # 根据作用域返回或创建实例
                scope = registration.scope

                # 单例作用域
                if scope == InjectionScope.SINGLETON:
                    # 如果已有实例，直接返回
                    if registration.instance is not None:
                        return cast(T, registration.instance)

                    # 否则创建实例
                    instance = self._create_instance_from_registration(
                        registration)
                    registration.instance = instance
                    return cast(T, instance)

                # 作用域依赖
                elif scope == InjectionScope.SCOPED:
                    # 检查是否有活动作用域
                    if self._current_scope_id is None:
                        raise ValueError(
                            f"尝试解析作用域依赖，但没有活动作用域: {resolution_id}")

                    # 查找或创建作用域实例
                    scope_dict = self._scoped_instances.setdefault(
                        self._current_scope_id, {})
                    type_dict = scope_dict.setdefault(interface_type, {})

                    if name in type_dict:
                        return cast(T, type_dict[name])

                    # 创建新实例
                    instance = self._create_instance_from_registration(
                        registration)
                    type_dict[name] = instance
                    return cast(T, instance)

                # 瞬态依赖
                elif scope == InjectionScope.TRANSIENT:
                    # 每次都创建新实例
                    return cast(T, self._create_instance_from_registration(
                        registration))

                else:
                    raise ValueError(f"未知的依赖作用域: {scope}")

            finally:
                # 从解析栈中移除
                self._resolution_stack.pop()

    def resolve_all(self, interface_type: Type[T]) -> List[T]:
        """
        解析某个接口的所有实现

        Args:
            interface_type: 接口类型

        Returns:
            List[T]: 所有实现的实例列表
        """
        with self._lock:
            if interface_type not in self._registrations:
                return []

            instances = []
            for name in self._registrations[interface_type]:
                try:
                    instance = self.resolve(interface_type, name)
                    instances.append(instance)
                except Exception as e:
                    logger.error(
                        f"解析依赖时出错: {interface_type.__name__}{f'({name})' if name else ''}, 错误: {e}")

            return instances

    def resolve_by_tag(self, tag: str) -> List[Any]:
        """
        按标签解析依赖

        Args:
            tag: 标签

        Returns:
            List[Any]: 带有指定标签的所有实例列表
        """
        with self._lock:
            instances = []

            # 遍历所有注册
            for interface_type in self._registrations:
                for name, registration in self._registrations[
                    interface_type].items():
                    if tag in registration.tags:
                        try:
                            instance = self.resolve(interface_type, name)
                            instances.append(instance)
                        except Exception as e:
                            logger.error(f"按标签解析依赖时出错: {tag}, "
                                         f"{interface_type.__name__}{f'({name})' if name else ''}, 错误: {e}")

            return instances

    def begin_scope(self, scope_id: Optional[str] = None) -> str:
        """
        开始新的依赖作用域

        Args:
            scope_id: 可选的作用域ID，如果为None则自动生成

        Returns:
            str: 作用域ID
        """
        with self._lock:
            # 生成作用域ID（如果未提供）
            if scope_id is None:
                import uuid
                scope_id = str(uuid.uuid4())

            # 设置当前作用域
            self._current_scope_id = scope_id

            # 确保作用域字典存在
            if scope_id not in self._scoped_instances:
                self._scoped_instances[scope_id] = {}

            logger.debug(f"开始新的依赖作用域: {scope_id}")
            return scope_id

    def end_scope(self, scope_id: Optional[str] = None) -> None:
        """
        结束依赖作用域

        Args:
            scope_id: 作用域ID，如果为None则使用当前作用域
        """
        with self._lock:
            # 确定作用域ID
            if scope_id is None:
                scope_id = self._current_scope_id

            if scope_id is None:
                logger.warning("尝试结束不存在的作用域")
                return

            # 清理作用域实例
            if scope_id in self._scoped_instances:
                # 释放资源
                for interface_type in self._scoped_instances[scope_id]:
                    for name, instance in self._scoped_instances[scope_id][
                        interface_type].items():
                        # 尝试调用释放资源方法
                        try:
                            if hasattr(instance, 'dispose') and callable(
                                    getattr(instance, 'dispose')):
                                instance.dispose()
                            elif hasattr(instance, 'close') and callable(
                                    getattr(instance, 'close')):
                                instance.close()
                            elif hasattr(instance, 'shutdown') and callable(
                                    getattr(instance, 'shutdown')):
                                instance.shutdown()
                        except Exception as e:
                            logger.warning(
                                f"释放作用域实例资源时出错: {type(instance).__name__}, 错误: {e}")

                # 移除作用域
                del self._scoped_instances[scope_id]

            # 如果是当前作用域，清除当前作用域ID
            if scope_id == self._current_scope_id:
                self._current_scope_id = None

            logger.debug(f"已结束依赖作用域: {scope_id}")

    def inject(self, instance: Any) -> None:
        """
        将依赖注入到现有实例

        Args:
            instance: 要注入依赖的实例
        """
        with self._lock:
            if instance is None:
                return

            # 获取类型提示
            type_hints = get_type_hints(type(instance))

            # 遍历所有成员
            for name, member_type in type_hints.items():
                # 检查是否可注入
                if name.startswith('_'):  # 跳过私有成员
                    continue

                # 检查是否已有值
                if hasattr(instance, name) and getattr(instance,
                                                       name) is not None:
                    continue

                # 尝试解析依赖
                try:
                    dependency = self.resolve(member_type)
                    setattr(instance, name, dependency)
                except Exception as e:
                    logger.debug(
                        f"注入依赖时出错: {type(instance).__name__}.{name}, 错误: {e}")

    def _create_instance_from_registration(self,
                                           registration: DependencyRegistration) -> Any:
        """
        从注册信息创建实例

        Args:
            registration: 依赖注册信息

        Returns:
            Any: 创建的实例

        Raises:
            ValueError: 创建实例时出错
        """
        # 如果有工厂函数，使用工厂创建
        if registration.factory is not None:
            # 处理工厂函数的依赖
            if registration.dependencies:
                kwargs = {}
                for param_name, dep_type in registration.dependencies.items():
                    kwargs[param_name] = self.resolve(dep_type)
                return registration.factory(**kwargs)
            else:
                return registration.factory()

        # 否则使用实现类型创建
        elif registration.implementation_type is not None:
            return self._create_instance(registration.implementation_type,
                                         registration.dependencies)

        # 没有可用的创建方式
        else:
            raise ValueError(
                f"无法创建实例: 注册信息缺少实现类型和工厂函数")

    def _create_instance(self, implementation_type: Type,
                         dependencies: Optional[
                             Dict[str, Any]] = None) -> Any:
        """
        创建指定类型的实例

        Args:
            implementation_type: 实现类型
            dependencies: 显式依赖字典

        Returns:
            Any: 创建的实例

        Raises:
            ValueError: 创建实例时出错
        """
        try:
            # 获取构造函数参数
            constructor = implementation_type.__init__
            sig = inspect.signature(constructor)

            # 准备构造函数参数
            kwargs = {}

            # 添加显式依赖
            if dependencies:
                for param_name, dep_type in dependencies.items():
                    if param_name in sig.parameters:
                        kwargs[param_name] = self.resolve(dep_type)

            # 自动解析其他参数
            for param_name, param in sig.parameters.items():
                # 跳过self参数
                if param_name == 'self':
                    continue

                # 跳过已经有值的参数
                if param_name in kwargs:
                    continue

                # 跳过可变参数
                if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD):
                    continue

                # 跳过没有类型注解的参数
                if param.annotation is inspect.Parameter.empty:
                    # 如果有默认值，使用默认值
                    if param.default is not inspect.Parameter.empty:
                        continue
                    # 否则抛出错误
                    else:
                        logger.warning(
                            f"参数没有类型注解且没有默认值: {implementation_type.__name__}.{param_name}")
                        continue

                # 尝试解析依赖
                try:
                    dependency = self.resolve(param.annotation)
                    kwargs[param_name] = dependency
                except Exception as e:
                    # 如果有默认值，使用默认值
                    if param.default is not inspect.Parameter.empty:
                        continue
                    # 否则抛出错误
                    else:
                        logger.warning(
                            f"无法解析参数依赖: {implementation_type.__name__}.{param_name}, 错误: {e}")

            # 创建实例
            instance = implementation_type(**kwargs)

            # 注入属性依赖
            self.inject(instance)

            return instance

        except Exception as e:
            logger.error(
                f"创建实例时出错: {implementation_type.__name__}, 错误: {e}")
            raise ValueError(
                f"创建实例时出错: {implementation_type.__name__}, 错误: {e}")

    def get_registration_count(self) -> int:
        """
        获取注册数量

        Returns:
            int: 注册数量
        """
        with self._lock:
            count = 0
            for interface_type in self._registrations:
                count += len(self._registrations[interface_type])
            return count

    def clear(self) -> None:
        """清除所有注册"""
        with self._lock:
            # 结束所有作用域
            for scope_id in list(self._scoped_instances.keys()):
                self.end_scope(scope_id)

            # 释放所有单例资源
            for interface_type in self._registrations:
                for name, registration in self._registrations[
                    interface_type].items():
                    if registration.instance is not None:
                        try:
                            if hasattr(registration.instance,
                                       'dispose') and callable(
                                    getattr(registration.instance,
                                            'dispose')):
                                registration.instance.dispose()
                            elif hasattr(registration.instance,
                                         'close') and callable(
                                    getattr(registration.instance,
                                            'close')):
                                registration.instance.close()
                            elif hasattr(registration.instance,
                                         'shutdown') and callable(
                                    getattr(registration.instance,
                                            'shutdown')):
                                registration.instance.shutdown()
                        except Exception as e:
                            logger.warning(
                                f"释放单例实例资源时出错: {type(registration.instance).__name__}, 错误: {e}")

            # 清除注册表
            self._registrations.clear()

            logger.info("依赖注入容器已清除所有注册")

# 便捷函数 - 获取依赖注入容器单例
def get_container() -> DependencyContainer:
    """获取依赖注入容器单例实例"""
    return DependencyContainer.get_instance()



    # 装饰器 - 自动注入
    def inject(cls):
        """
        类装饰器 - 自动注入所有依赖

        用法:
            @inject
            class MyService:
                # 将自动注入类型为Logger的实例
                logger: Logger

                def __init__(self, db: Database):
                    # 构造函数参数也会自动注入
                    self.db = db
        """
        # 保存原始构造函数
        original_init = cls.__init__

        # 创建新的构造函数
        def __init__(self, *args, **kwargs):
            # 调用原始构造函数
            original_init(self, *args, **kwargs)

            # 注入依赖
            container = get_container()
            container.inject(self)

        # 替换构造函数
        cls.__init__ = __init__

        return cls

