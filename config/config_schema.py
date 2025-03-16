#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置模式定义模块 - 定义系统配置的结构、类型和验证规则

该模块提供了配置项的结构定义、默认值和验证规则，确保配置的一致性和有效性。
它定义了各种配置验证器和配置模式类，用于验证和描述系统配置。
"""

import re
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar, \
    Set, cast

logger = logging.getLogger(__name__)

# 用于类型提示的类型变量
T = TypeVar('T')


class ConfigType(Enum):
    """配置值类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    OBJECT = "object"
    ARRAY = "array"


class ConfigSchema:
    """配置项模式类，定义配置项的类型、约束和验证规则"""

    def __init__(
            self,
            key: str,
            type_: ConfigType,
            default_value: Any = None,
            required: bool = False,
            description: str = "",
            constraints: Optional[Dict[str, Any]] = None,
            children: Optional[List['ConfigSchema']] = None
    ):
        """初始化配置模式

        Args:
            key: 配置键
            type_: 配置值类型
            default_value: 默认值
            required: 是否必需
            description: 描述文本
            constraints: 约束条件
            children: 子配置项（用于对象类型）
        """
        self.key = key
        self.type = type_
        self.default_value = default_value
        self.required = required
        self.description = description
        self.constraints = constraints or {}
        self.children = children or []

    def validate(self, value: Any) -> bool:
        """验证配置值是否符合模式

        Args:
            value: 要验证的值

        Returns:
            bool: 是否有效
        """
        # 检查必需值
        if self.required and value is None:
            logger.error(f"Required config key '{self.key}' is missing")
            return False

        # 空值检查
        if value is None:
            return True

        # 按类型验证
        valid = False
        try:
            if self.type == ConfigType.STRING:
                valid = self._validate_string(value)
            elif self.type == ConfigType.INTEGER:
                valid = self._validate_integer(value)
            elif self.type == ConfigType.FLOAT:
                valid = self._validate_float(value)
            elif self.type == ConfigType.BOOLEAN:
                valid = self._validate_boolean(value)
            elif self.type == ConfigType.ENUM:
                valid = self._validate_enum(value)
            elif self.type == ConfigType.OBJECT:
                valid = self._validate_object(value)
            elif self.type == ConfigType.ARRAY:
                valid = self._validate_array(value)
            else:
                logger.warning(
                    f"Unknown config type '{self.type}' for key '{self.key}'")
                return False
        except Exception as e:
            logger.error(f"Error validating config '{self.key}': {str(e)}")
            return False

        if not valid:
            logger.warning(
                f"Config validation failed for key '{self.key}' with value {value}")

        return valid

    def _validate_string(self, value: Any) -> bool:
        """验证字符串类型配置值"""
        if not isinstance(value, str):
            return False

        # 长度验证
        min_length = self.constraints.get('min_length')
        if min_length is not None and len(value) < min_length:
            return False

        max_length = self.constraints.get('max_length')
        if max_length is not None and len(value) > max_length:
            return False

        # 正则表达式验证
        pattern = self.constraints.get('pattern')
        if pattern is not None and not re.match(pattern, value):
            return False

        return True

    def _validate_integer(self, value: Any) -> bool:
        """验证整数类型配置值"""
        if not isinstance(value, int) or isinstance(value, bool):
            return False

        # 范围验证
        min_value = self.constraints.get('min')
        if min_value is not None and value < min_value:
            return False

        max_value = self.constraints.get('max')
        if max_value is not None and value > max_value:
            return False

        return True

    def _validate_float(self, value: Any) -> bool:
        """验证浮点数类型配置值"""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False

        # 范围验证
        min_value = self.constraints.get('min')
        if min_value is not None and value < min_value:
            return False

        max_value = self.constraints.get('max')
        if max_value is not None and value > max_value:
            return False

        return True

    def _validate_boolean(self, value: Any) -> bool:
        """验证布尔类型配置值"""
        return isinstance(value, bool)

    def _validate_enum(self, value: Any) -> bool:
        """验证枚举类型配置值"""
        allowed_values = self.constraints.get('values', [])
        return value in allowed_values

    def _validate_object(self, value: Any) -> bool:
        """验证对象类型配置值"""
        if not isinstance(value, dict):
            return False

        # 如果没有子模式，只验证是否为字典
        if not self.children:
            return True

        # 检查每个子模式
        all_valid = True
        for child_schema in self.children:
            # 如果是必需的键但不存在
            if child_schema.required and child_schema.key not in value:
                logger.warning(
                    f"Required key '{child_schema.key}' missing in object config '{self.key}'")
                all_valid = False
                continue

            # 如果键存在，验证其值
            if child_schema.key in value:
                child_value = value[child_schema.key]
                if not child_schema.validate(child_value):
                    all_valid = False

        return all_valid

    def _validate_array(self, value: Any) -> bool:
        """验证数组类型配置值"""
        if not isinstance(value, list):
            return False

        # 长度验证
        min_items = self.constraints.get('min_items')
        if min_items is not None and len(value) < min_items:
            return False

        max_items = self.constraints.get('max_items')
        if max_items is not None and len(value) > max_items:
            return False

        # 如果有项目类型约束，验证每个项目
        item_schema = self.constraints.get('item_schema')
        if item_schema is not None:
            for item in value:
                if not item_schema.validate(item):
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """将模式转换为字典表示

        Returns:
            Dict[str, Any]: 模式的字典表示
        """
        result = {
            'key': self.key,
            'type': self.type.value,
            'default_value': self.default_value,
            'required': self.required,
            'description': self.description,
            'constraints': self.constraints.copy()
        }

        if self.children:
            result['children'] = [child.to_dict() for child in self.children]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigSchema':
        """从字典创建模式

        Args:
            data: 模式字典

        Returns:
            ConfigSchema: 创建的模式实例
        """
        type_value = data.get('type', 'string')
        type_enum = ConfigType(type_value)

        children_data = data.get('children', [])
        children = [cls.from_dict(child) for child in children_data]

        return cls(
            key=data['key'],
            type_=type_enum,
            default_value=data.get('default_value'),
            required=data.get('required', False),
            description=data.get('description', ''),
            constraints=data.get('constraints', {}),
            children=children
        )


class ConfigSchemaRegistry:
    """配置模式注册表，管理所有配置项的模式"""

    _instance = None

    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(ConfigSchemaRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化配置模式注册表"""
        if self._initialized:
            return

        self._schemas: Dict[str, ConfigSchema] = {}
        self._categories: Dict[str, List[str]] = {}
        self._defaults: Dict[str, Any] = {}
        self._initialized = True

    def register_schema(self, schema: ConfigSchema,
                        category: Optional[str] = None) -> None:
        """注册配置模式

        Args:
            schema: 配置模式
            category: 配置分类
        """
        self._schemas[schema.key] = schema

        # 保存默认值
        if schema.default_value is not None:
            self._defaults[schema.key] = schema.default_value

        # 关联到分类
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(schema.key)

        # 注册子模式
        for child in schema.children:
            child_key = f"{schema.key}.{child.key}"
            child_schema = ConfigSchema(
                key=child_key,
                type_=child.type,
                default_value=child.default_value,
                required=child.required,
                description=child.description,
                constraints=child.constraints,
                children=child.children
            )
            self.register_schema(child_schema, category)

    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """获取配置模式

        Args:
            key: 配置键

        Returns:
            ConfigSchema: 配置模式，如果不存在则返回None
        """
        return self._schemas.get(key)

    def get_all_schemas(self) -> Dict[str, ConfigSchema]:
        """获取所有配置模式

        Returns:
            Dict[str, ConfigSchema]: 配置键到模式的映射
        """
        return self._schemas.copy()

    def get_categories(self) -> Dict[str, List[str]]:
        """获取所有配置分类

        Returns:
            Dict[str, List[str]]: 分类名称到配置键列表的映射
        """
        return self._categories.copy()

    def get_default_values(self) -> Dict[str, Any]:
        """获取所有默认值

        Returns:
            Dict[str, Any]: 配置键到默认值的映射
        """
        return self._defaults.copy()

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """验证整个配置对象

        Args:
            config: 要验证的配置对象

        Returns:
            Dict[str, List[str]]: 验证错误，键为配置键，值为错误消息列表
        """
        errors: Dict[str, List[str]] = {}

        # 检查每个模式
        for key, schema in self._schemas.items():
            value = get_nested_value(config, key)

            # 如果是必需的但不存在
            if schema.required and value is None:
                if key not in errors:
                    errors[key] = []
                errors[key].append(f"Required config key '{key}' is missing")
                continue

            # 如果值存在，验证其有效性
            if value is not None and not schema.validate(value):
                if key not in errors:
                    errors[key] = []
                errors[key].append(
                    f"Invalid value for config key '{key}': {value}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """将所有模式转换为字典表示

        Returns:
            Dict[str, Any]: 模式的字典表示
        """
        return {
            'schemas': {k: v.to_dict() for k, v in self._schemas.items()},
            'categories': self._categories.copy(),
            'defaults': self._defaults.copy()
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载模式

        Args:
            data: 模式字典
        """
        # 清除现有数据
        self._schemas.clear()
        self._categories.clear()
        self._defaults.clear()

        # 加载模式
        schemas_data = data.get('schemas', {})
        for key, schema_data in schemas_data.items():
            schema = ConfigSchema.from_dict(schema_data)
            self._schemas[key] = schema

        # 加载分类
        self._categories = data.get('categories', {}).copy()

        # 加载默认值
        self._defaults = data.get('defaults', {}).copy()


def get_nested_value(data: Dict[str, Any], key: str) -> Any:
    """获取嵌套字典中的值

    Args:
        data: 数据字典
        key: 点号分隔的键路径

    Returns:
        任何值，如果键不存在则返回None
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


def set_nested_value(data: Dict[str, Any], key: str, value: Any) -> None:
    """设置嵌套字典中的值

    Args:
        data: 数据字典
        key: 点号分隔的键路径
        value: 要设置的值
    """
    if '.' not in key:
        data[key] = value
        return

    parts = key.split('.')
    current = data

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def create_schema_from_value(key: str, value: Any,
                             description: str = "") -> ConfigSchema:
    """根据值创建配置模式

    Args:
        key: 配置键
        value: 值
        description: 描述

    Returns:
        ConfigSchema: 创建的模式
    """
    if isinstance(value, str):
        return ConfigSchema(key, ConfigType.STRING, value,
                            description=description)
    elif isinstance(value, bool):
        return ConfigSchema(key, ConfigType.BOOLEAN, value,
                            description=description)
    elif isinstance(value, int):
        return ConfigSchema(key, ConfigType.INTEGER, value,
                            description=description)
    elif isinstance(value, float):
        return ConfigSchema(key, ConfigType.FLOAT, value,
                            description=description)
    elif isinstance(value, list):
        return ConfigSchema(key, ConfigType.ARRAY, value,
                            description=description)
    elif isinstance(value, dict):
        children = []
        for child_key, child_value in value.items():
            child_schema = create_schema_from_value(child_key, child_value)
            children.append(child_schema)
        return ConfigSchema(key, ConfigType.OBJECT, value,
                            description=description, children=children)
    else:
        # 默认为字符串类型
        return ConfigSchema(key, ConfigType.STRING, str(value),
                            description=description)


def create_schemas_from_dict(config_dict: Dict[str, Any]) -> List[ConfigSchema]:
    """从字典创建配置模式列表

    Args:
        config_dict: 配置字典

    Returns:
        List[ConfigSchema]: 配置模式列表
    """
    schemas = []
    for key, value in config_dict.items():
        schema = create_schema_from_value(key, value)
        schemas.append(schema)
    return schemas


# 初始化默认配置模式
def init_default_schemas() -> Dict[str, ConfigSchema]:
    """初始化默认配置模式

    Returns:
        Dict[str, ConfigSchema]: 配置键到模式的映射
    """
    schemas = {}

    # 应用基础配置
    app_schema = ConfigSchema(
        key="app",
        type_=ConfigType.OBJECT,
        description="应用程序基础配置",
        children=[
            ConfigSchema(
                key="name",
                type_=ConfigType.STRING,
                default_value="人体姿态跟踪系统",
                required=True,
                description="应用程序名称"
            ),
            ConfigSchema(
                key="version",
                type_=ConfigType.STRING,
                default_value="1.0.0",
                required=True,
                description="应用程序版本"
            )
        ]
    )
    schemas["app"] = app_schema

    # 检测器配置
    detector_schema = ConfigSchema(
        key="detector",
        type_=ConfigType.OBJECT,
        description="人体姿态检测器配置",
        children=[
            ConfigSchema(
                key="use_mediapipe",
                type_=ConfigType.BOOLEAN,
                default_value=False,
                description="是否使用MediaPipe增强"
            ),
            ConfigSchema(
                key="performance_mode",
                type_=ConfigType.ENUM,
                default_value="balanced",
                description="性能模式设置",
                constraints={
                    "values": ["fast", "balanced", "accurate"]
                }
            ),
            ConfigSchema(
                key="confidence_threshold",
                type_=ConfigType.FLOAT,
                default_value=0.5,
                description="检测置信度阈值",
                constraints={
                    "min": 0.0,
                    "max": 1.0
                }
            )
        ]
    )
    schemas["detector"] = detector_schema

    # UI配置
    ui_schema = ConfigSchema(
        key="ui",
        type_=ConfigType.OBJECT,
        description="用户界面配置",
        children=[
            ConfigSchema(
                key="show_debug_info",
                type_=ConfigType.BOOLEAN,
                default_value=False,
                description="是否显示调试信息"
            ),
            ConfigSchema(
                key="camera_width",
                type_=ConfigType.INTEGER,
                default_value=640,
                description="相机视图宽度",
                constraints={
                    "min": 320,
                    "max": 1920
                }
            ),
            ConfigSchema(
                key="camera_height",
                type_=ConfigType.INTEGER,
                default_value=480,
                description="相机视图高度",
                constraints={
                    "min": 240,
                    "max": 1080
                }
            ),
            ConfigSchema(
                key="theme",
                type_=ConfigType.ENUM,
                default_value="light",
                description="界面主题",
                constraints={
                    "values": ["light", "dark", "system"]
                }
            )
        ]
    )
    schemas["ui"] = ui_schema

    # 系统配置
    system_schema = ConfigSchema(
        key="system",
        type_=ConfigType.OBJECT,
        description="系统配置",
        children=[
            ConfigSchema(
                key="log_level",
                type_=ConfigType.ENUM,
                default_value="info",
                description="日志级别",
                constraints={
                    "values": ["debug", "info", "warning", "error", "critical"]
                }
            ),
            ConfigSchema(
                key="async_mode",
                type_=ConfigType.BOOLEAN,
                default_value=False,
                description="是否使用异步处理模式"
            ),
            ConfigSchema(
                key="max_threads",
                type_=ConfigType.INTEGER,
                default_value=4,
                description="最大线程数",
                constraints={
                    "min": 1,
                    "max": 16
                }
            ),
            ConfigSchema(
                key="cache_size",
                type_=ConfigType.INTEGER,
                default_value=100,
                description="缓存大小",
                constraints={
                    "min": 10,
                    "max": 1000
                }
            )
        ]
    )
    schemas["system"] = system_schema

    # 映射器配置
    mapper_schema = ConfigSchema(
        key="mapper",
        type_=ConfigType.OBJECT,
        description="位置映射器配置",
        children=[
            ConfigSchema(
                key="smooth_factor",
                type_=ConfigType.FLOAT,
                default_value=0.5,
                description="平滑因子",
                constraints={
                    "min": 0.0,
                    "max": 1.0
                }
            ),
            ConfigSchema(
                key="room_width",
                type_=ConfigType.INTEGER,
                default_value=800,
                description="虚拟房间宽度",
                constraints={
                    "min": 400,
                    "max": 2000
                }
            ),
            ConfigSchema(
                key="room_height",
                type_=ConfigType.INTEGER,
                default_value=600,
                description="虚拟房间高度",
                constraints={
                    "min": 300,
                    "max": 1500
                }
            )
        ]
    )
    schemas["mapper"] = mapper_schema

    # 动作识别器配置
    recognizer_schema = ConfigSchema(
        key="recognizer",
        type_=ConfigType.OBJECT,
        description="动作识别器配置",
        children=[
            ConfigSchema(
                key="history_size",
                type_=ConfigType.INTEGER,
                default_value=10,
                description="姿态历史大小",
                constraints={
                    "min": 5,
                    "max": 50
                }
            ),
            ConfigSchema(
                key="use_dtw",
                type_=ConfigType.BOOLEAN,
                default_value=False,
                description="是否使用DTW算法"
            )
        ]
    )
    schemas["recognizer"] = recognizer_schema

    return schemas


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 初始化模式注册表
    registry = ConfigSchemaRegistry()

    # 初始化默认模式
    default_schemas = init_default_schemas()
    for key, schema in default_schemas.items():
        registry.register_schema(schema, category=key.split('.')[
            0] if '.' in key else key)

    # 使用示例：验证配置
    test_config = {
        "app": {
            "name": "测试应用",
            "version": "1.0.0"
        },
        "detector": {
            "use_mediapipe": True,
            "performance_mode": "invalid_mode",  # 这将导致验证错误
            "confidence_threshold": 0.7
        },
        "ui": {
            "show_debug_info": True,
            "camera_width": 800,
            "camera_height": 600,
            "theme": "dark"
        }
    }

    # 验证配置
    errors = registry.validate_config(test_config)
    if errors:
        print("配置验证错误:")
        for key, messages in errors.items():
            for msg in messages:
                print(f"  - {msg}")
    else:
        print("配置验证通过")

    # 获取默认值
    defaults = registry.get_default_values()
    print("\n默认配置:")
    print(json.dumps(defaults, indent=2, ensure_ascii=False))

    # 获取分类
    categories = registry.get_categories()
    print("\n配置分类:")
    for category, keys in categories.items():
        print(f"  {category}: {', '.join(keys)}")
