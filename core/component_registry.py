# -*- coding: utf-8 -*-
"""
组件注册表模块 - 管理组件类型和实例
提供组件注册、查找、配置和状态管理功能
"""

import logging
import threading
import time
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Type, TypeVar, Callable, \
    Union, Generic, Tuple
from dataclasses import dataclass, field

from .component_lifecycle import LifecycleState, LifecycleManager
from core.dependency_container import get_container, DependencyContainer
from .event_bus import get_event_bus, EventBus
from .event_models import EventType, EventPriority

# 设置日志
logger = logging.getLogger("ComponentRegistry")

# 类型变量
T = TypeVar('T')
Component = TypeVar('Component')


class ComponentCategory(Enum):
    """组件类别枚举"""
    CORE = "core"  # 核心组件
    SERVICE = "service"  # 服务组件
    PROCESSOR = "processor"  # 处理器组件
    DETECTOR = "detector"  # 检测器组件
    RECOGNIZER = "recognizer"  # 识别器组件
    MAPPER = "mapper"  # 映射器组件
    UI = "ui"  # UI组件
    UTILITY = "utility"  # 工具组件
    EXTERNAL = "external"  # 外部组件
    CUSTOM = "custom"  # 自定义组件


@dataclass
class ComponentInfo:
    """组件信息类"""
    # 基本信息
    component_id: str  # 组件ID
    component_type: Type  # 组件类型
    component_instance: Any  # 组件实例

    # 分类信息
    category: ComponentCategory  # 组件类别
    tags: Set[str] = field(default_factory=set)  # 组件标签

    # 组件状态
    lifecycle_manager: Optional[LifecycleManager] = None  # 生命周期管理器

    # 配置信息
    config_section: Optional[str] = None  # 配置节名称
    config_schema: Optional[Dict[str, Any]] = None  # 配置模式

    # 版本信息
    version: str = "1.0.0"  # 组件版本

    # 时间信息
    registration_time: float = field(default_factory=time.time)  # 注册时间
    last_state_change_time: float = field(default_factory=time.time)  # 最后状态变更时间

    # 统计信息
    event_count: int = 0  # 处理的事件数
    error_count: int = 0  # 错误数

    def get_current_state(self) -> Optional[LifecycleState]:
        """获取组件当前状态"""
        if self.lifecycle_manager:
            return self.lifecycle_manager.get_current_state()
        return None

    def get_uptime(self) -> float:
        """获取组件运行时间（秒）"""
        if self.lifecycle_manager:
            return self.lifecycle_manager.get_total_uptime()
        return 0.0

    def is_running(self) -> bool:
        """检查组件是否运行中"""
        state = self.get_current_state()
        return state == LifecycleState.RUNNING if state else False

    def is_error(self) -> bool:
        """检查组件是否处于错误状态"""
        state = self.get_current_state()
        return state == LifecycleState.ERROR if state else False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        state = self.get_current_state()
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.__name__,
            'category': self.category.value,
            'tags': list(self.tags),
            'state': state.name if state else None,
            'version': self.version,
            'uptime': self.get_uptime(),
            'registration_time': self.registration_time,
            'event_count': self.event_count,
            'error_count': self.error_count,
            'is_running': self.is_running()
        }


class ComponentRegistry:
    """
    组件注册表 - 管理系统中的所有组件
    提供组件注册、查找、生命周期管理和配置管理功能
    """

    _instance = None  # 单例实例

    @classmethod
    def get_instance(cls) -> 'ComponentRegistry':
        """获取ComponentRegistry单例实例"""
        if cls._instance is None:
            cls._instance = ComponentRegistry()
        return cls._instance

    def __init__(self):
        """初始化组件注册表"""
        if ComponentRegistry._instance is not None:
            logger.warning(
                "ComponentRegistry是单例类，请使用get_instance()获取实例")
            return

        # 组件信息字典 {component_id: ComponentInfo}
        self._components: Dict[str, ComponentInfo] = {}

        # 类型映射 {组件类型: {component_id}}
        self._type_map: Dict[Type, Set[str]] = {}

        # 类别映射 {组件类别: {component_id}}
        self._category_map: Dict[ComponentCategory, Set[str]] = {}

        # 标签映射 {标签: {component_id}}
        self._tag_map: Dict[str, Set[str]] = {}

        # 获取依赖注入容器
        self._container = get_container()

        # 获取事件总线
        self._event_bus = get_event_bus()

        # 线程锁
        self._lock = threading.RLock()

        logger.info("组件注册表已初始化")

    def register_component(self,
                           component_instance: Any,
                           component_id: Optional[str] = None,
                           category: ComponentCategory = ComponentCategory.CUSTOM,
                           tags: Optional[Set[str]] = None,
                           config_section: Optional[str] = None,
                           config_schema: Optional[Dict[str, Any]] = None,
                           version: str = "1.0.0",
                           auto_manage_lifecycle: bool = True) -> str:
        """
        注册组件

        Args:
            component_instance: 组件实例
            component_id: 组件ID，如果为None则使用类名+实例ID
            category: 组件类别
            tags: 组件标签集合
            config_section: 配置节名称
            config_schema: 配置模式
            version: 组件版本
            auto_manage_lifecycle: 是否自动管理生命周期

        Returns:
            str: 组件ID
        """
        with self._lock:
            # 组件类型
            component_type = type(component_instance)

            # 生成组件ID（如果未提供）
            if component_id is None:
                component_id = f"{component_type.__name__}_{id(component_instance)}"

            # 检查是否已注册
            if component_id in self._components:
                logger.warning(f"组件ID '{component_id}' 已存在，将覆盖现有注册")
                self.unregister_component(component_id)

            # 创建生命周期管理器（如果需要）
            lifecycle_manager = None
            if auto_manage_lifecycle:
                if hasattr(component_instance, 'lifecycle_manager'):
                    # 组件已有生命周期管理器
                    lifecycle_manager = component_instance.lifecycle_manager
                else:
                    # 创建新的生命周期管理器
                    from .component_lifecycle import create_lifecycle_manager
                    lifecycle_manager = create_lifecycle_manager(component_id,
                                                                 component_type.__name__)

                    # 如果组件实现了生命周期接口，绑定生命周期管理器
                    if hasattr(component_instance, 'set_lifecycle_manager'):
                        component_instance.set_lifecycle_manager(
                            lifecycle_manager)

            # 创建组件信息
            component_info = ComponentInfo(
                component_id=component_id,
                component_type=component_type,
                component_instance=component_instance,
                category=category,
                tags=tags or set(),
                lifecycle_manager=lifecycle_manager,
                config_section=config_section,
                config_schema=config_schema,
                version=version
            )

            # 添加到注册表
            self._components[component_id] = component_info

            # 更新类型映射
            if component_type not in self._type_map:
                self._type_map[component_type] = set()
            self._type_map[component_type].add(component_id)

            # 更新类别映射
            if category not in self._category_map:
                self._category_map[category] = set()
            self._category_map[category].add(component_id)

            # 更新标签映射
            for tag in component_info.tags:
                if tag not in self._tag_map:
                    self._tag_map[tag] = set()
                self._tag_map[tag].add(component_id)

            # 在依赖注入容器中注册
            self._container.register_instance(component_type,
                                              component_instance,
                                              name=component_id)

            # 发布组件注册事件
            self._event_bus.create_and_publish(
                event_type=EventType.COMPONENT_REGISTERED,
                data={
                    'component_id': component_id,
                    'component_type': component_type.__name__,
                    'category': category.value,
                    'tags': list(component_info.tags),
                    'timestamp': time.time()
                },
                source_id="component_registry",
                priority=EventPriority.NORMAL
            )

            logger.info(
                f"已注册组件: {component_id}, 类型: {component_type.__name__}, 类别: {category.value}")

            return component_id

    def unregister_component(self, component_id: str) -> bool:
        """
        取消注册组件

        Args:
            component_id: 组件ID

        Returns:
            bool: 是否成功取消注册
        """
        with self._lock:
            if component_id not in self._components:
                logger.warning(f"尝试取消注册不存在的组件: {component_id}")
                return False

            # 获取组件信息
            component_info = self._components[component_id]

            # 停止并销毁组件（如果有生命周期管理器）
            if component_info.lifecycle_manager:
                try:
                    current_state = component_info.lifecycle_manager.get_current_state()

                    # 如果组件正在运行，先停止它
                    if current_state == LifecycleState.RUNNING:
                        component_info.lifecycle_manager.transition_to(
                            LifecycleState.STOPPING)
                        component_info.lifecycle_manager.transition_to(
                            LifecycleState.STOPPED)

                    # 销毁组件
                    if current_state != LifecycleState.DESTROYED:
                        component_info.lifecycle_manager.transition_to(
                            LifecycleState.DESTROYING)
                        component_info.lifecycle_manager.transition_to(
                            LifecycleState.DESTROYED)

                except Exception as e:
                    logger.error(
                        f"停止/销毁组件时出错: {component_id}, 错误: {e}")

            # 从类型映射中移除
            component_type = component_info.component_type
            if component_type in self._type_map:
                self._type_map[component_type].discard(component_id)
                if not self._type_map[component_type]:
                    del self._type_map[component_type]

            # 从类别映射中移除
            category = component_info.category
            if category in self._category_map:
                self._category_map[category].discard(component_id)
                if not self._category_map[category]:
                    del self._category_map[category]

            # 从标签映射中移除
            for tag in component_info.tags:
                if tag in self._tag_map:
                    self._tag_map[tag].discard(component_id)
                    if not self._tag_map[tag]:
                        del self._tag_map[tag]

            # 从组件字典中移除
            del self._components[component_id]

            # 发布组件注销事件
            self._event_bus.create_and_publish(
                event_type=EventType.COMPONENT_STATE_CHANGED,
                data={
                    'component_id': component_id,
                    'component_type': component_type.__name__,
                    'from_state': 'REGISTERED',
                    'to_state': 'UNREGISTERED',
                    'timestamp': time.time()
                },
                source_id="component_registry",
                priority=EventPriority.NORMAL
            )

            logger.info(f"已取消注册组件: {component_id}")

            return True

    def get_component(self, component_id: str) -> Optional[Any]:
        """
        获取组件实例

        Args:
            component_id: 组件ID

        Returns:
            Any or None: 组件实例，如果不存在则返回None
        """
        with self._lock:
            if component_id not in self._components:
                return None

            return self._components[component_id].component_instance

    def get_component_info(self, component_id: str) -> Optional[ComponentInfo]:
        """
        获取组件信息

        Args:
            component_id: 组件ID

        Returns:
            ComponentInfo or None: 组件信息，如果不存在则返回None
        """
        with self._lock:
            return self._components.get(component_id)

    def get_components_by_type(self, component_type: Type[T]) -> List[T]:
        """
        按类型获取组件实例

        Args:
            component_type: 组件类型

        Returns:
            List[T]: 匹配类型的组件实例列表
        """
        with self._lock:
            result = []

            # 检查类型映射
            if component_type in self._type_map:
                for component_id in self._type_map[component_type]:
                    component = self.get_component(component_id)
                    if component:
                        result.append(component)

            return result

    def get_components_by_category(self, category: ComponentCategory) -> List[
        Any]:
        """
        按类别获取组件实例

        Args:
            category: 组件类别

        Returns:
            List[Any]: 匹配类别的组件实例列表
        """
        with self._lock:
            result = []

            # 检查类别映射
            if category in self._category_map:
                for component_id in self._category_map[category]:
                    component = self.get_component(component_id)
                    if component:
                        result.append(component)

            return result

    def get_components_by_tag(self, tag: str) -> List[Any]:
        """
        按标签获取组件实例

        Args:
            tag: 组件标签

        Returns:
            List[Any]: 带有指定标签的组件实例列表
        """
        with self._lock:
            result = []

            # 检查标签映射
            if tag in self._tag_map:
                for component_id in self._tag_map[tag]:
                    component = self.get_component(component_id)
                    if component:
                        result.append(component)

            return result

    def get_all_components(self) -> Dict[str, Any]:
        """
        获取所有组件实例

        Returns:
            Dict[str, Any]: 组件ID到实例的映射
        """
        with self._lock:
            return {component_id: info.component_instance
                    for component_id, info in self._components.items()}

    def get_all_component_info(self) -> Dict[str, ComponentInfo]:
        """
        获取所有组件信息

        Returns:
            Dict[str, ComponentInfo]: 组件ID到信息的映射
        """
        with self._lock:
            return dict(self._components)

    def get_component_count(self) -> int:
        """
        获取组件总数

        Returns:
            int: 组件总数
        """
        with self._lock:
            return len(self._components)

    def start_component(self, component_id: str) -> bool:
        """
        启动组件

        Args:
            component_id: 组件ID

        Returns:
            bool: 是否成功启动
        """
        with self._lock:
            component_info = self.get_component_info(component_id)
            if not component_info:
                logger.warning(f"尝试启动不存在的组件: {component_id}")
                return False

            # 检查是否有生命周期管理器
            if not component_info.lifecycle_manager:
                logger.warning(f"组件没有生命周期管理器: {component_id}")
                return False

            try:
                # 获取当前状态
                current_state = component_info.lifecycle_manager.get_current_state()

                # 如果组件未初始化，先初始化
                if current_state in [LifecycleState.REGISTERED,
                                     LifecycleState.UNREGISTERED]:
                    component_info.lifecycle_manager.transition_to(
                        LifecycleState.INITIALIZING)
                    component_info.lifecycle_manager.transition_to(
                        LifecycleState.INITIALIZED)

                # 启动组件
                component_info.lifecycle_manager.transition_to(
                    LifecycleState.STARTING)
                component_info.lifecycle_manager.transition_to(
                    LifecycleState.RUNNING)

                logger.info(f"已启动组件: {component_id}")
                return True

            except Exception as e:
                logger.error(f"启动组件时出错: {component_id}, 错误: {e}")
                return False

    def stop_component(self, component_id: str) -> bool:
        """
        停止组件

        Args:
            component_id: 组件ID

        Returns:
            bool: 是否成功停止
        """
        with self._lock:
            component_info = self.get_component_info(component_id)
            if not component_info:
                logger.warning(f"尝试停止不存在的组件: {component_id}")
                return False

            # 检查是否有生命周期管理器
            if not component_info.lifecycle_manager:
                logger.warning(f"组件没有生命周期管理器: {component_id}")
                return False

            try:
                # 停止组件
                component_info.lifecycle_manager.transition_to(
                    LifecycleState.STOPPING)
                component_info.lifecycle_manager.transition_to(
                    LifecycleState.STOPPED)

                logger.info(f"已停止组件: {component_id}")
                return True

            except Exception as e:
                logger.error(f"停止组件时出错: {component_id}, 错误: {e}")
                return False

        def start_all_components(self, categories: Optional[
            List[ComponentCategory]] = None) -> Dict[str, bool]:
            """
            启动所有组件

            Args:
                categories: 要启动的组件类别列表，None表示所有类别

            Returns:
                Dict[str, bool]: 组件ID到启动结果的映射
            """
            with self._lock:
                result = {}

                # 确定要启动的组件ID
                component_ids = []
                if categories:
                    for category in categories:
                        if category in self._category_map:
                            component_ids.extend(self._category_map[category])
                else:
                    component_ids = list(self._components.keys())

                # 按优先级顺序启动组件
                # 优先级顺序: CORE > SERVICE > 其他
                priority_order = [
                    ComponentCategory.CORE,
                    ComponentCategory.SERVICE
                ]

                # 首先启动优先级组件
                for category in priority_order:
                    for component_id in component_ids:
                        if component_id in self._components and \
                                self._components[
                                    component_id].category == category:
                            result[component_id] = self.start_component(
                                component_id)

                # 然后启动其他组件
                for component_id in component_ids:
                    if component_id not in result:
                        result[component_id] = self.start_component(
                            component_id)

                return result

        def stop_all_components(self, categories: Optional[
            List[ComponentCategory]] = None) -> Dict[str, bool]:
            """
            停止所有组件

            Args:
                categories: 要停止的组件类别列表，None表示所有类别

            Returns:
                Dict[str, bool]: 组件ID到停止结果的映射
            """
            with self._lock:
                result = {}

                # 确定要停止的组件ID
                component_ids = []
                if categories:
                    for category in categories:
                        if category in self._category_map:
                            component_ids.extend(self._category_map[category])
                else:
                    component_ids = list(self._components.keys())

                # 按优先级相反顺序停止组件
                # 优先级顺序: 其他 > SERVICE > CORE
                priority_order = [
                    ComponentCategory.CORE,
                    ComponentCategory.SERVICE
                ]

                # 首先停止非优先级组件
                for component_id in component_ids:
                    if component_id in self._components and self._components[
                        component_id].category not in priority_order:
                        result[component_id] = self.stop_component(component_id)

                # 然后按相反优先级顺序停止优先级组件
                for category in reversed(priority_order):
                    for component_id in component_ids:
                        if component_id in self._components and \
                                self._components[
                                    component_id].category == category:
                            result[component_id] = self.stop_component(
                                component_id)

                return result

        def update_component_config(self, component_id: str,
                                    config: Dict[str, Any]) -> bool:
            """
            更新组件配置

            Args:
                component_id: 组件ID
                config: 新配置

            Returns:
                bool: 是否成功更新配置
            """
            with self._lock:
                component_info = self.get_component_info(component_id)
                if not component_info:
                    logger.warning(f"尝试更新不存在的组件配置: {component_id}")
                    return False

                try:
                    # 如果组件支持配置更新接口
                    component = component_info.component_instance
                    if hasattr(component, 'update_config') and callable(
                            getattr(component, 'update_config')):
                        component.update_config(config)
                        logger.info(f"已更新组件配置: {component_id}")
                        return True
                    else:
                        logger.warning(f"组件不支持配置更新: {component_id}")
                        return False

                except Exception as e:
                    logger.error(
                        f"更新组件配置时出错: {component_id}, 错误: {e}")
                    return False

    def get_component_stats(self, component_id: str) -> Optional[
        Dict[str, Any]]:
        """
        获取组件统计信息

        Args:
            component_id: 组件ID

        Returns:
            Dict or None: 组件统计信息，如果不存在则返回None
        """
        with self._lock:
            component_info = self.get_component_info(component_id)
            if not component_info:
                return None

            # 获取基本信息
            stats = component_info.to_dict()

            # 如果组件支持统计接口，获取更多信息
            component = component_info.component_instance
            if hasattr(component, 'get_stats') and callable(
                    getattr(component, 'get_stats')):
                try:
                    component_stats = component.get_stats()
                    if isinstance(component_stats, dict):
                        stats.update(component_stats)
                except Exception as e:
                    logger.error(
                        f"获取组件统计时出错: {component_id}, 错误: {e}")

            return stats

    def get_system_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            Dict: 系统统计信息
        """
        with self._lock:
            # 基本统计信息
            stats = {
                'total_components': len(self._components),
                'running_components': 0,
                'error_components': 0,
                'component_categories': {},
                'uptime': 0.0,  # 平均运行时间
                'component_list': []
            }

            # 收集组件信息
            for component_id, component_info in self._components.items():
                # 更新状态计数
                if component_info.is_running():
                    stats['running_components'] += 1
                if component_info.is_error():
                    stats['error_components'] += 1

                # 更新类别计数
                category = component_info.category.value
                if category in stats['component_categories']:
                    stats['component_categories'][category] += 1
                else:
                    stats['component_categories'][category] = 1

                # 累加运行时间
                stats['uptime'] += component_info.get_uptime()

                # 添加组件基本信息
                stats['component_list'].append({
                    'id': component_id,
                    'type': component_info.component_type.__name__,
                    'category': category,
                    'state': component_info.get_current_state().name if component_info.get_current_state() else None,
                    'is_running': component_info.is_running(),
                    'is_error': component_info.is_error()
                })

            # 计算平均运行时间
            if stats['total_components'] > 0:
                stats['uptime'] /= stats['total_components']

            return stats

    def update_component_tag(self, component_id: str, tag: str,
                             add: bool = True) -> bool:
        """
        更新组件标签

        Args:
            component_id: 组件ID
            tag: 标签
            add: True表示添加标签，False表示移除标签

        Returns:
            bool: 是否成功更新标签
        """
        with self._lock:
            component_info = self.get_component_info(component_id)
            if not component_info:
                logger.warning(f"尝试更新不存在的组件标签: {component_id}")
                return False

            if add:
                # 添加标签
                if tag not in component_info.tags:
                    component_info.tags.add(tag)

                    # 更新标签映射
                    if tag not in self._tag_map:
                        self._tag_map[tag] = set()
                    self._tag_map[tag].add(component_id)

                    logger.debug(f"已为组件 {component_id} 添加标签: {tag}")
                    return True
                return False
            else:
                # 移除标签
                if tag in component_info.tags:
                    component_info.tags.discard(tag)

                    # 更新标签映射
                    if tag in self._tag_map:
                        self._tag_map[tag].discard(component_id)
                        if not self._tag_map[tag]:
                            del self._tag_map[tag]

                    logger.debug(f"已从组件 {component_id} 移除标签: {tag}")
                    return True
                return False

    def check_dependencies(self, component_id: str) -> Dict[str, bool]:
        """
        检查组件依赖

        Args:
            component_id: 组件ID

        Returns:
            Dict[str, bool]: 依赖名称到是否满足的映射
        """
        with self._lock:
            component_info = self.get_component_info(component_id)
            if not component_info:
                logger.warning(f"尝试检查不存在的组件依赖: {component_id}")
                return {}

            result = {}

            # 如果组件支持依赖检查接口
            component = component_info.component_instance
            if hasattr(component, 'get_dependencies') and callable(
                    getattr(component, 'get_dependencies')):
                try:
                    dependencies = component.get_dependencies()
                    for dep_name, dep_type in dependencies.items():
                        # 检查依赖是否满足
                        try:
                            self._container.resolve(dep_type)
                            result[dep_name] = True
                        except Exception:
                            result[dep_name] = False
                except Exception as e:
                    logger.error(
                        f"获取组件依赖时出错: {component_id}, 错误: {e}")

            return result

    def clear(self):
        """清除所有组件"""
        with self._lock:
            # 先停止所有组件
            self.stop_all_components()

            # 取消注册所有组件
            for component_id in list(self._components.keys()):
                self.unregister_component(component_id)

            # 清除各种映射
            self._type_map.clear()
            self._category_map.clear()
            self._tag_map.clear()

            logger.info("组件注册表已清除所有组件")

# 便捷函数 - 获取组件注册表单例
def get_component_registry() -> ComponentRegistry:
    """获取组件注册表单例实例"""
    return ComponentRegistry.get_instance()
