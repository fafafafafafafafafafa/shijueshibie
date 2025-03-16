#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器模块 - 系统配置的中央管理系统

该模块提供了一个统一的配置管理器，负责加载、验证、存储和分发系统配置。
它实现了单例模式，确保整个应用程序中只有一个配置管理器实例。
"""

import os
import json
import logging
import threading
import copy
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# 配置更改回调的类型定义
ConfigChangeCallback = Callable[[str, Any, Any], None]


class ConfigManager:
    """配置管理器类 - 负责管理所有系统配置

    该类实现了单例模式，确保整个应用程序中只有一个配置管理器实例。
    它提供了加载、获取、设置和监视配置的方法，并支持配置更改通知。
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config_dir: str = "config",
                 default_config_file: str = "default_config.json"):
        """初始化配置管理器

        Args:
            config_dir: 配置文件目录
            default_config_file: 默认配置文件名
        """
        # 避免重复初始化
        with self._lock:
            if self._initialized:
                return

            self._config_dir = config_dir
            self._default_config_file = default_config_file
            self._config_file_path = os.path.join(config_dir,
                                                  default_config_file)

            # 当前配置存储
            self._config: Dict[str, Any] = {}

            # 默认配置存储 - 不应随意修改
            self._default_config: Dict[str, Any] = {}

            # 配置观察者，格式: {config_key: {observer_id: callback_function}}
            self._observers: Dict[str, Dict[str, ConfigChangeCallback]] = {}

            # 配置验证函数映射
            self._validators: Dict[str, Callable[[Any], bool]] = {}

            # 配置描述映射
            self._config_descriptions: Dict[str, str] = {}

            # 配置分类
            self._config_categories: Dict[str, List[str]] = {}

            # 配置变更历史
            self._config_history: List[Dict[str, Any]] = []
            self._max_history_size = 100

            # 标记为已初始化
            self._initialized = True

            # 确保配置目录存在
            self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """确保配置目录存在"""
        if not os.path.exists(self._config_dir):
            try:
                os.makedirs(self._config_dir)
                logger.info(f"Created config directory: {self._config_dir}")
            except OSError as e:
                logger.error(f"Failed to create config directory: {e}")

    def load_default_config(self, default_config: Dict[str, Any]) -> None:
        """加载默认配置

        Args:
            default_config: 默认配置字典
        """
        with self._lock:
            # 使用深拷贝确保默认配置不会被其他操作修改
            self._default_config = copy.deepcopy(default_config)

            # 如果当前配置为空，使用默认配置初始化
            if not self._config:
                self._config = copy.deepcopy(default_config)
                logger.info("Initialized with default configuration")

    def load_config_from_file(self, file_path: Optional[str] = None) -> bool:
        """从文件加载配置

        Args:
            file_path: 配置文件路径，如果为None则使用默认路径

        Returns:
            bool: 加载成功返回True，否则返回False
        """
        path = file_path or self._config_file_path

        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}")
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # 使用加载的配置更新当前配置，但不修改默认配置
            with self._lock:
                for key, value in self._flatten_dict(loaded_config).items():
                    old_value = self.get(key)
                    self._set_nested_value(self._config, key, value,
                                           notify=False)
                    # 记录历史但不通知
                    self._add_to_history(key, old_value, value)

            logger.info(f"Successfully loaded configuration from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {str(e)}")
            return False

    def save_config_to_file(self, file_path: Optional[str] = None) -> bool:
        """将当前配置保存到文件

        Args:
            file_path: 配置文件路径，如果为None则使用默认路径

        Returns:
            bool: 保存成功返回True，否则返回False
        """
        path = file_path or self._config_file_path

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)

            logger.info(f"Successfully saved configuration to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {str(e)}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键
            default: 默认值，如果键不存在则返回此值

        Returns:
            配置值或默认值
        """
        with self._lock:
            return self._get_nested_value(self._config, key, default)

    def _get_nested_value(self, config_dict: Dict[str, Any], key: str,
                          default: Any = None) -> Any:
        """获取嵌套字典中的值

        Args:
            config_dict: 配置字典
            key: 点号分隔的键
            default: 默认值

        Returns:
            配置值或默认值
        """
        # 支持点号分隔的嵌套键
        if '.' in key:
            parts = key.split('.')
            current = config_dict
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current

        return config_dict.get(key, default)

    def set(self, key: str, value: Any, notify: bool = True,
            validate: bool = True) -> bool:
        """设置配置值

        Args:
            key: 配置键
            value: 配置值
            notify: 是否通知观察者
            validate: 是否验证值

        Returns:
            bool: 设置成功返回True，否则返回False
        """
        with self._lock:
            # 如果启用验证且键有关联的验证器
            if validate and key in self._validators:
                validator = self._validators[key]
                if not validator(value):
                    logger.warning(
                        f"Validation failed for config key '{key}' with value {value}")
                    return False

            # 获取旧值（用于通知观察者）
            old_value = self.get(key)

            # 设置新值
            self._set_nested_value(self._config, key, value, notify, old_value)

            return True

    def _set_nested_value(self, config_dict: Dict[str, Any], key: str,
                          value: Any,
                          notify: bool = True, old_value: Any = None) -> None:
        """设置嵌套字典中的值

        Args:
            config_dict: 配置字典
            key: 点号分隔的键
            value: 要设置的值
            notify: 是否通知观察者
            old_value: 旧值，如果不提供则从配置中获取
        """
        # 支持点号分隔的嵌套键
        if '.' in key:
            parts = key.split('.')
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # 如果未提供旧值，获取当前值
            if old_value is None:
                old_value = self._get_nested_value(config_dict, key)

            # 设置新值
            current[parts[-1]] = value
        else:
            # 如果未提供旧值，获取当前值
            if old_value is None:
                old_value = config_dict.get(key)

            # 设置新值
            config_dict[key] = value

        # 添加到历史记录
        self._add_to_history(key, old_value, value)

        # 通知观察者
        if notify and old_value != value:
            self._notify_observers(key, old_value, value)

    def _flatten_dict(self, nested_dict: Dict[str, Any], prefix: str = "") -> \
    Dict[str, Any]:
        """将嵌套字典扁平化为点号分隔的键值对

        Args:
            nested_dict: 嵌套字典
            prefix: 键前缀

        Returns:
            Dict[str, Any]: 扁平化的字典
        """
        items = {}
        for key, value in nested_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(self._flatten_dict(value, new_key))
            else:
                items[new_key] = value
        return items

    def _add_to_history(self, key: str, old_value: Any, new_value: Any) -> None:
        """将配置更改添加到历史记录

        Args:
            key: 配置键
            old_value: 旧值
            new_value: 新值
        """
        timestamp = datetime.now().isoformat()

        change_record = {
            'key': key,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': timestamp
        }

        self._config_history.append(change_record)

        # 限制历史记录大小
        if len(self._config_history) > self._max_history_size:
            self._config_history = self._config_history[
                                   -self._max_history_size:]

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取配置变更历史

        Args:
            limit: 返回的最大记录数

        Returns:
            List[Dict]: 配置变更历史记录列表
        """
        with self._lock:
            if limit is None:
                return self._config_history.copy()
            return self._config_history[-limit:].copy()

    def register_validator(self, key: str, validator: Callable[[Any], bool],
                           description: Optional[str] = None) -> None:
        """为配置键注册验证器

        Args:
            key: 配置键
            validator: 验证函数，接受值并返回布尔值
            description: 配置项描述
        """
        with self._lock:
            self._validators[key] = validator
            if description:
                self._config_descriptions[key] = description

    def register_category(self, category: str, keys: List[str]) -> None:
        """注册配置分类

        Args:
            category: 分类名称
            keys: 属于该分类的配置键列表
        """
        with self._lock:
            if category not in self._config_categories:
                self._config_categories[category] = []
            self._config_categories[category].extend(keys)

    def get_categories(self) -> Dict[str, List[str]]:
        """获取所有配置分类

        Returns:
            Dict[str, List[str]]: 分类名称到配置键列表的映射
        """
        with self._lock:
            return self._config_categories.copy()

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置

        Returns:
            Dict[str, Any]: 当前配置的完整副本
        """
        with self._lock:
            return copy.deepcopy(self._config)

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置

        Returns:
            Dict[str, Any]: 默认配置的完整副本
        """
        with self._lock:
            return copy.deepcopy(self._default_config)

    def reset_to_defaults(self, notify: bool = True) -> None:
        """将配置重置为默认值

        Args:
            notify: 是否通知观察者
        """
        with self._lock:
            # 记录旧值用于通知
            old_values = {}
            if notify:
                # 只记录与默认值不同的配置项
                for key, value in self._flatten_dict(self._config).items():
                    default_value = self._get_nested_value(self._default_config,
                                                           key)
                    if value != default_value:
                        old_values[key] = value

            # 清空当前配置
            self._config.clear()

            # 使用深拷贝恢复默认配置，确保不会意外修改默认配置
            self._config = copy.deepcopy(self._default_config)

            # 通知变更
            if notify:
                for key, old_value in old_values.items():
                    new_value = self._get_nested_value(self._config, key)
                    self._notify_observers(key, old_value, new_value)

            logger.info("Configuration reset to defaults")

    def subscribe(self, observer_id: str, key: str,
                  callback: ConfigChangeCallback) -> None:
        """订阅配置更改通知

        Args:
            observer_id: 观察者ID
            key: 配置键
            callback: 当配置更改时调用的回调函数
        """
        with self._lock:
            if key not in self._observers:
                self._observers[key] = {}

            self._observers[key][observer_id] = callback

    def unsubscribe(self, observer_id: str, key: Optional[str] = None) -> None:
        """取消订阅配置更改通知

        Args:
            observer_id: 观察者ID
            key: 配置键，如果为None则取消所有键的订阅
        """
        with self._lock:
            if key is None:
                # 取消所有键的订阅
                for k in list(self._observers.keys()):
                    if observer_id in self._observers[k]:
                        del self._observers[k][observer_id]
            elif key in self._observers and observer_id in self._observers[key]:
                del self._observers[key][observer_id]

    def _notify_observers(self, key: str, old_value: Any,
                          new_value: Any) -> None:
        """通知配置键的观察者

        Args:
            key: 配置键
            old_value: 旧值
            new_value: 新值
        """
        # 复制观察者字典以避免在迭代过程中修改
        observers = {}
        with self._lock:
            if key in self._observers:
                observers = self._observers[key].copy()

        # 通知观察者
        for observer_id, callback in observers.items():
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(
                    f"Error in config observer {observer_id} for key {key}: {str(e)}")

    def add_config_description(self, key: str, description: str) -> None:
        """添加配置项描述

        Args:
            key: 配置键
            description: 描述文本
        """
        with self._lock:
            self._config_descriptions[key] = description

    def get_config_description(self, key: str) -> Optional[str]:
        """获取配置项描述

        Args:
            key: 配置键

        Returns:
            str: 描述文本，如果不存在则返回None
        """
        with self._lock:
            return self._config_descriptions.get(key)

    def get_all_descriptions(self) -> Dict[str, str]:
        """获取所有配置项描述

        Returns:
            Dict[str, str]: 配置键到描述的映射
        """
        with self._lock:
            return self._config_descriptions.copy()


# 以下是使用ConfigManager的示例代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 获取ConfigManager实例
    config_manager = ConfigManager()

    # 加载默认配置
    default_config = {
        "app": {
            "name": "人体姿态跟踪系统",
            "version": "1.0.0"
        },
        "detector": {
            "use_mediapipe": False,
            "performance_mode": "balanced"
        },
        "ui": {
            "show_debug_info": False,
            "camera_width": 640,
            "camera_height": 480
        },
        "system": {
            "log_level": "info",
            "async_mode": False
        }
    }
    config_manager.load_default_config(default_config)

    # 注册验证器
    config_manager.register_validator(
        "detector.performance_mode",
        lambda mode: mode in ["fast", "balanced", "accurate"],
        "检测器性能模式，可选值：fast, balanced, accurate"
    )

    # 注册分类
    config_manager.register_category("ui",
                                     ["ui.show_debug_info", "ui.camera_width",
                                      "ui.camera_height"])
    config_manager.register_category("detector", ["detector.use_mediapipe",
                                                  "detector.performance_mode"])


    # 订阅配置更改
    def on_config_changed(key, old_value, new_value):
        print(f"配置已更改: {key} = {new_value} (原值: {old_value})")


    config_manager.subscribe("example_observer", "detector.performance_mode",
                             on_config_changed)

    # 设置配置
    config_manager.set("detector.performance_mode", "accurate")

    # 获取配置
    performance_mode = config_manager.get("detector.performance_mode")
    print(f"当前性能模式: {performance_mode}")

    # 保存配置
    config_manager.save_config_to_file()

    # 修改配置并保存到新文件
    config_manager.set("app.name", "测试应用")
    config_manager.save_config_to_file("test_config.json")

    # 重置配置
    config_manager.reset_to_defaults()
    print(f"重置后的应用名称: {config_manager.get('app.name')}")
