#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置系统模块 - 统一的配置系统入口

该模块整合了配置管理器、配置模式、配置变更通知和配置文件监视器，
提供统一的配置系统接口，使应用程序能够便捷地使用配置功能。
"""

import os
import logging
import threading
from typing import Any, Dict, List, Optional, Callable, Set

# 导入配置系统组件
from config.config_manager import ConfigManager
from config.config_schema import ConfigSchemaRegistry, ConfigSchema, ConfigType, \
    init_default_schemas
from config.config_change_notifier import ConfigChangeNotifier, \
    NotificationMode, get_notifier, create_handler
from config.config_persistence import ConfigPersistenceManager, ConfigFormat
from config.config_watcher import ConfigWatcher, WatchMode

logger = logging.getLogger(__name__)

# 回调函数类型
ConfigChangeCallback = Callable[[str, Any, Any], None]


class ConfigSystem:
    """统一的配置系统入口，整合所有配置组件"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigSystem, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
            self,
            config_dir: str = "config",
            default_config_file: str = "config.json",
            watch_interval: float = 1.0,
            enable_file_watching: bool = True,
            enable_schema_validation: bool = True,
            enable_hot_reload: bool = True
    ):
        """初始化配置系统

        Args:
            config_dir: 配置目录
            default_config_file: 默认配置文件
            watch_interval: 文件监视间隔（秒）
            enable_file_watching: 是否启用文件监视
            enable_schema_validation: 是否启用模式验证
            enable_hot_reload: 是否启用热重载
        """
        # 避免重复初始化
        with self._lock:
            if self._initialized:
                return

            self.config_dir = config_dir
            self.default_config_file = default_config_file
            self.watch_interval = watch_interval
            self.enable_file_watching = enable_file_watching
            self.enable_schema_validation = enable_schema_validation
            self.enable_hot_reload = enable_hot_reload

            # 创建组件
            self._config_manager = ConfigManager(config_dir,
                                                 default_config_file)
            self._schema_registry = ConfigSchemaRegistry()
            self._notifier = get_notifier(enable_batch_mode=True)
            self._persistence = ConfigPersistenceManager(config_dir)

            if enable_file_watching:
                self._watcher = ConfigWatcher(
                    mode=WatchMode.POLL,
                    interval=watch_interval,
                    auto_reload=enable_hot_reload
                )
            else:
                self._watcher = None

            # 已加载的配置文件
            self._loaded_files = set()

            # 初始化标志
            self._initialized = True

            # 如果启用了热重载和文件监视，设置文件变更处理
            if enable_hot_reload and enable_file_watching:
                self._setup_hot_reload()

    def _setup_hot_reload(self) -> None:
        """设置配置热重载"""
        # 添加全局文件变更回调
        self._watcher.add_global_callback(self._on_config_file_changed)

        # 如果监视器尚未运行，启动它
        if not self._watcher._running:
            self._watcher.start()

        logger.info("Config hot reload enabled")

    def _on_config_file_changed(self, file_path: str) -> None:
        """配置文件变更回调

        Args:
            file_path: 变更的文件路径
        """
        logger.info(f"Detected change in config file: {file_path}")

        try:
            # 只处理已加载的文件
            if file_path not in self._loaded_files:
                logger.debug(f"Ignoring change for unloaded file: {file_path}")
                return

            # 加载新配置
            config = self._persistence.load(file_path)

            # 如果启用了模式验证，验证配置
            if self.enable_schema_validation:
                errors = self._schema_registry.validate_config(config)
                if errors:
                    error_messages = []
                    for key, messages in errors.items():
                        for msg in messages:
                            error_messages.append(f"{key}: {msg}")

                    logger.error(
                        f"Invalid config in {file_path}: {', '.join(error_messages)}")
                    return

            # 应用配置，记录变更
            old_values = {}
            for key, value in self._flatten_config(config).items():
                old_value = self._config_manager.get(key)
                old_values[key] = old_value
                self._config_manager.set(key, value, notify=False)

            # 通知变更
            for key, new_value in self._flatten_config(config).items():
                old_value = old_values.get(key)
                self._notifier.notify(key, old_value, new_value,
                                      source=file_path)

            logger.info(f"Reloaded config from {file_path}")
        except Exception as e:
            logger.error(f"Error reloading config from {file_path}: {str(e)}")

    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[
        str, Any]:
        """将嵌套配置扁平化为点号分隔的键

        Args:
            config: 配置字典
            prefix: 键前缀

        Returns:
            Dict[str, Any]: 扁平化配置
        """
        result = {}

        for key, value in config.items():
            flat_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # 递归处理嵌套字典
                result.update(self._flatten_config(value, flat_key))
            else:
                # 添加叶节点
                result[flat_key] = value

        return result

    def init_default_schemas(self) -> None:
        """初始化默认配置模式"""
        # 获取默认模式
        default_schemas = init_default_schemas()

        # 注册模式
        for key, schema in default_schemas.items():
            self._schema_registry.register_schema(schema, category=key)

        # 更新通知器的分类
        self._notifier.set_categories(self._schema_registry.get_categories())

        logger.info("Initialized default config schemas")

    def load_config(
            self,
            file_path: Optional[str] = None,
            merge: bool = True
    ) -> Dict[str, Any]:
        """加载配置文件

        Args:
            file_path: 配置文件路径，如果为None则使用默认路径
            merge: 是否合并到现有配置，而不是替换

        Returns:
            Dict[str, Any]: 加载的配置

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 配置格式不支持或解析错误
        """
        # 如果未指定文件路径，使用默认路径
        if file_path is None:
            file_path = os.path.join(self.config_dir, self.default_config_file)

        try:
            # 加载配置
            config = self._persistence.load(file_path)

            # 记录已加载的文件
            self._loaded_files.add(file_path)

            # 如果启用了文件监视，监视这个文件
            if self.enable_file_watching and self._watcher:
                self._watcher.watch_file(file_path)

            # 如果启用了模式验证，验证配置
            if self.enable_schema_validation:
                errors = self._schema_registry.validate_config(config)
                if errors:
                    error_messages = []
                    for key, messages in errors.items():
                        for msg in messages:
                            error_messages.append(f"{key}: {msg}")

                    logger.error(
                        f"Invalid config in {file_path}: {', '.join(error_messages)}")
                    raise ValueError(
                        f"Invalid configuration: {', '.join(error_messages)}")

            # 应用配置
            if merge:
                # 合并到现有配置
                current_config = self._config_manager.get_all_config()
                merged_config = self._persistence.merge(current_config, config)

                # 只设置变更的键
                for key, value in self._flatten_config(config).items():
                    old_value = self._config_manager.get(key)
                    # 如果值不同，设置并通知
                    if old_value != value:
                        self._config_manager.set(key, value)
            else:
                # 完全替换配置
                # 获取当前所有配置以便对比变更
                old_config = self._config_manager.get_all_config()

                # 设置新配置
                flattened_config = self._flatten_config(config)

                # 先收集所有变更，然后一次性应用
                changes = {}
                for key, value in flattened_config.items():
                    old_value = self._get_nested_value(old_config, key)
                    if old_value != value:
                        changes[key] = (old_value, value)

                # 应用所有变更
                for key, (old_value, value) in changes.items():
                    self._config_manager.set(key, value, notify=True)

            logger.info(f"Loaded config from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {str(e)}")
            raise

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """获取嵌套字典中的值

        Args:
            data: 数据字典
            key: 点号分隔的键路径

        Returns:
            Any: 值，如果键不存在则返回None
        """
        if '.' not in key:
            return data.get(key)

        parts = key.split('.')
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def save_config(
            self,
            file_path: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            create_backup: bool = True
    ) -> bool:
        """保存配置到文件

        Args:
            file_path: 文件路径，如果为None则使用默认路径
            config: 要保存的配置，如果为None则保存当前配置
            create_backup: 是否创建备份

        Returns:
            bool: 是否成功保存
        """
        # 如果未指定文件路径，使用默认路径
        if file_path is None:
            file_path = os.path.join(self.config_dir, self.default_config_file)

        # 如果未指定配置，使用当前配置
        if config is None:
            config = self._config_manager.get_all_config()

        try:
            # 保存配置
            success = self._persistence.save(config, file_path, create_backup)

            if success:
                # 记录已加载的文件
                self._loaded_files.add(file_path)

                # 如果启用了文件监视，监视这个文件
                if self.enable_file_watching and self._watcher:
                    self._watcher.watch_file(file_path)

                logger.info(f"Saved config to {file_path}")

            return success
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {str(e)}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键
            default: 默认值，如果键不存在则返回此值

        Returns:
            Any: 配置值或默认值
        """
        return self._config_manager.get(key, default)

    def set(self, key: str, value: Any, persist: bool = False) -> bool:
        """设置配置值

        Args:
            key: 配置键
            value: 配置值
            persist: 是否持久化到文件

        Returns:
            bool: 是否成功设置
        """
        result = self._config_manager.set(key, value)

        # 如果需要持久化且设置成功
        if persist and result:
            # 保存到默认配置文件
            self.save_config()

        return result

    def subscribe(
            self,
            subscriber_id: str,
            key: str,
            callback: ConfigChangeCallback,
            subscription_type: str = "exact",
            **kwargs
    ) -> None:
        """订阅配置变更

        Args:
            subscriber_id: 订阅者ID
            key: 配置键或模式
            callback: 回调函数
            subscription_type: 订阅类型（'exact', 'prefix', 'pattern', 'category', 'all'）
            **kwargs: 其他订阅选项
        """
        # 转换回调函数格式
        handler = create_handler(subscriber_id, callback)

        # 根据订阅类型选择合适的订阅方法
        if subscription_type == "exact":
            self._notifier.subscribe_exact(subscriber_id, key, handler,
                                           **kwargs)
        elif subscription_type == "prefix":
            self._notifier.subscribe_prefix(subscriber_id, key, handler,
                                            **kwargs)
        elif subscription_type == "pattern":
            self._notifier.subscribe_pattern(subscriber_id, key, handler,
                                             **kwargs)
        elif subscription_type == "category":
            self._notifier.subscribe_category(subscriber_id, key, handler,
                                              **kwargs)
        elif subscription_type == "all":
            self._notifier.subscribe_all(subscriber_id, handler, **kwargs)
        else:
            raise ValueError(f"Unknown subscription type: {subscription_type}")

        logger.debug(
            f"Added config subscriber {subscriber_id} for {subscription_type} '{key}'")

    def unsubscribe(self, subscriber_id: str) -> bool:
        """取消订阅配置变更

        Args:
            subscriber_id: 订阅者ID

        Returns:
            bool: 是否成功取消订阅
        """
        return self._notifier.unsubscribe(subscriber_id)

    def register_schema(self, schema: ConfigSchema,
                        category: Optional[str] = None) -> None:
        """注册配置模式

        Args:
            schema: 配置模式
            category: 配置分类
        """
        self._schema_registry.register_schema(schema, category)

        # 更新通知器的分类
        self._notifier.set_categories(self._schema_registry.get_categories())

    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """获取配置模式

        Args:
            key: 配置键

        Returns:
            Optional[ConfigSchema]: 配置模式，如果不存在则返回None
        """
        return self._schema_registry.get_schema(key)

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """验证配置

        Args:
            config: 要验证的配置

        Returns:
            Dict[str, List[str]]: 验证错误
        """
        return self._schema_registry.validate_config(config)

    def watch_config_dir(self, pattern: Optional[str] = None) -> bool:
        """监视配置目录

        Args:
            pattern: 文件名模式

        Returns:
            bool: 是否成功添加监视
        """
        if not self.enable_file_watching or not self._watcher:
            logger.warning("File watching is disabled")
            return False

        return self._watcher.watch_directory(
            self.config_dir,
            pattern=pattern,
            recursive=False
        )

    def bind_component_config(
            self,
            component: Any,
            component_name: str,
            config_prefix: str,
            handler_method: Optional[str] = None
    ) -> None:
        """绑定组件的配置变更处理

        Args:
            component: 组件实例
            component_name: 组件名称
            config_prefix: 配置前缀
            handler_method: 处理方法名，如果为None，则尝试使用on_config_changed
        """
        # 确定处理方法
        method_name = handler_method or 'on_config_changed'
        if not hasattr(component, method_name) or not callable(
                getattr(component, method_name)):
            logger.warning(
                f"Component {component_name} does not have method {method_name}")
            return

        handler = getattr(component, method_name)

        # 创建订阅者ID
        subscriber_id = f"{component_name}_{config_prefix}"

        # 订阅前缀匹配的配置变更
        self.subscribe(
            subscriber_id=subscriber_id,
            key=config_prefix,
            callback=handler,
            subscription_type="prefix"
        )

        logger.debug(
            f"Bound component {component_name} to config prefix {config_prefix}")

    def reset_to_defaults(self, persist: bool = False) -> None:
        """将配置重置为默认值

        Args:
            persist: 是否持久化到文件
        """
        # 获取默认值
        defaults = self._schema_registry.get_default_values()

        # 应用默认值
        for key, value in defaults.items():
            self._config_manager.set(key, value)

        # 如果需要持久化
        if persist:
            self.save_config()

        logger.info("Reset configuration to defaults")

    def get_loaded_files(self) -> Set[str]:
        """获取已加载的配置文件

        Returns:
            Set[str]: 文件路径集合
        """
        return self._loaded_files.copy()

    def shutdown(self) -> None:
        """关闭配置系统，清理资源"""
        logger.info("Shutting down config system")

        # 停止文件监视
        if self._watcher:
            self._watcher.stop()

        # 关闭通知器
        self._notifier.shutdown()


# 获取配置系统实例的工厂函数
def get_config_system(
        config_dir: str = "config",
        default_config_file: str = "config.json",
        watch_interval: float = 1.0,
        enable_file_watching: bool = True,
        enable_schema_validation: bool = True,
        enable_hot_reload: bool = True
) -> ConfigSystem:
    """获取配置系统实例

    Args:
        config_dir: 配置目录
        default_config_file: 默认配置文件
        watch_interval: 文件监视间隔（秒）
        enable_file_watching: 是否启用文件监视
        enable_schema_validation: 是否启用模式验证
        enable_hot_reload: 是否启用热重载

    Returns:
        ConfigSystem: 配置系统实例
    """
    return ConfigSystem(
        config_dir=config_dir,
        default_config_file=default_config_file,
        watch_interval=watch_interval,
        enable_file_watching=enable_file_watching,
        enable_schema_validation=enable_schema_validation,
        enable_hot_reload=enable_hot_reload
    )


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 获取配置系统实例
    config_system = get_config_system(
        config_dir="./config_test",
        enable_hot_reload=True
    )

    # 初始化默认模式
    config_system.init_default_schemas()


    # 定义一些处理函数
    def handle_ui_config(key, old_value, new_value):
        print(f"[UI] Config changed: {key} = {new_value} (was: {old_value})")


    def handle_detector_config(key, old_value, new_value):
        print(
            f"[Detector] Config changed: {key} = {new_value} (was: {old_value})")


    def handle_any_config(key, old_value, new_value):
        print(f"[Any] Config changed: {key} = {new_value} (was: {old_value})")


    # 订阅配置变更
    config_system.subscribe("ui_handler", "ui", handle_ui_config,
                            subscription_type="prefix")
    config_system.subscribe("detector_handler", "detector",
                            handle_detector_config, subscription_type="prefix")
    config_system.subscribe("any_handler", "", handle_any_config,
                            subscription_type="all")

    # 设置一些配置值
    config_system.set("app.name", "测试应用")
    config_system.set("ui.theme", "dark")
    config_system.set("detector.performance_mode", "accurate")

    # 保存配置
    config_system.save_config()

    # 监视配置目录
    config_system.watch_config_dir(pattern=r"\.json$")

    print("Configuration system initialized")
    print("Try modifying the config file to test hot reload")
    print("Press Enter to exit...")

    try:
        input()
    finally:
        # 关闭配置系统
        config_system.shutdown()
