# -*- coding: utf-8 -*-
"""
插件配置管理 - 负责插件配置的验证和管理
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("PluginConfig")


def validate_plugin_config(config: Dict[str, Any], schema: Dict[str, Any]) -> \
Tuple[bool, Optional[str]]:
    """
    验证插件配置是否符合模式

    Args:
        config: 要验证的配置
        schema: 配置模式

    Returns:
        Tuple[bool, Optional[str]]: (是否有效, 错误信息)
    """
    try:
        # 检查必填字段
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in config:
                return False, f"缺少必填字段: {field}"

        # 检查字段类型
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in config:
                field_type = field_schema.get('type')
                if field_type:
                    # 基本类型检查
                    if field_type == 'string' and not isinstance(config[field],
                                                                 str):
                        return False, f"字段 {field} 应为字符串类型"
                    elif field_type == 'number' and not isinstance(
                            config[field], (int, float)):
                        return False, f"字段 {field} 应为数字类型"
                    elif field_type == 'integer' and not isinstance(
                            config[field], int):
                        return False, f"字段 {field} 应为整数类型"
                    elif field_type == 'boolean' and not isinstance(
                            config[field], bool):
                        return False, f"字段 {field} 应为布尔类型"
                    elif field_type == 'array' and not isinstance(config[field],
                                                                  list):
                        return False, f"字段 {field} 应为数组类型"
                    elif field_type == 'object' and not isinstance(
                            config[field], dict):
                        return False, f"字段 {field} 应为对象类型"

                # 检查枚举值
                enum = field_schema.get('enum')
                if enum and config[field] not in enum:
                    return False, f"字段 {field} 的值应为以下之一: {', '.join(map(str, enum))}"

                # 检查数值范围
                if isinstance(config[field], (int, float)):
                    minimum = field_schema.get('minimum')
                    maximum = field_schema.get('maximum')

                    if minimum is not None and config[field] < minimum:
                        return False, f"字段 {field} 的值不应小于 {minimum}"

                    if maximum is not None and config[field] > maximum:
                        return False, f"字段 {field} 的值不应大于 {maximum}"

        return True, None
    except Exception as e:
        logger.error(f"验证配置时出错: {e}")
        return False, f"验证出错: {e}"


def merge_plugin_configs(base_config: Dict[str, Any],
                         override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并插件配置

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        Dict[str, Any]: 合并后的配置
    """
    # 创建基础配置的副本
    result = base_config.copy()

    # 递归合并配置
    for key, value in override_config.items():
        if (key in result and isinstance(result[key], dict)
                and isinstance(value, dict)):
            # 递归合并嵌套字典
            result[key] = merge_plugin_configs(result[key], value)
        else:
            # 直接覆盖或添加值
            result[key] = value

    return result


def load_plugin_config(config_file: str) -> Dict[str, Any]:
    """
    从文件加载插件配置

    Args:
        config_file: 配置文件路径

    Returns:
        Dict[str, Any]: 加载的配置，加载失败返回空字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {config_file}, 错误: {e}")
        return {}


def save_plugin_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    保存插件配置到文件

    Args:
        config: 要保存的配置
        config_file: 配置文件路径

    Returns:
        bool: 是否成功保存
    """
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {config_file}, 错误: {e}")
        return False


def get_default_config(plugin_id: str) -> Dict[str, Any]:
    """
    获取插件的默认配置

    Args:
        plugin_id: 插件ID

    Returns:
        Dict[str, Any]: 默认配置
    """
    from .plugin_registry import get_plugin_info

    # 获取插件信息
    plugin_info = get_plugin_info(plugin_id)
    if not plugin_info:
        logger.warning(f"未找到插件: {plugin_id}")
        return {}

    # 获取配置模式
    schema = plugin_info.config_schema
    default_config = {}

    # 提取默认值
    properties = schema.get('properties', {})
    for field, field_schema in properties.items():
        if 'default' in field_schema:
            default_config[field] = field_schema['default']

    return default_config
