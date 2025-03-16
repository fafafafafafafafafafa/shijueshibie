# -*- coding: utf-8 -*-
"""
插件系统核心模块 - 提供插件系统的核心功能
"""

# 导出核心组件
from .plugin_interface import PluginBase, PluginInfo, PluginType
from .plugin_system import get_plugin_system
from .plugin_registry import init_registry, register_plugin, get_plugins, get_plugin_by_id
from .plugin_loader import discover_plugins, load_plugin, discover_and_load_plugins
from .plugin_config import validate_plugin_config, merge_plugin_configs
from .pipeline_builder import build_pipeline, validate_pipeline

__all__ = [
    'PluginBase', 'PluginInfo', 'PluginType',
    'get_plugin_system',
    'init_registry', 'register_plugin', 'get_plugins', 'get_plugin_by_id',
    'discover_plugins', 'load_plugin', 'discover_and_load_plugins',
    'validate_plugin_config', 'merge_plugin_configs',
    'build_pipeline', 'validate_pipeline'
]
