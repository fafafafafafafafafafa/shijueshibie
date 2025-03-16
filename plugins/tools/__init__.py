# -*- coding: utf-8 -*-
"""
工具插件包 - 包含各种工具和实用插件

此包提供了多种不同的工具插件，包括:
- 调试工具插件

每个工具插件都实现了标准的 ToolPluginInterface 接口，
可以无缝集成到插件系统中。
"""

from plugins.core.plugin_interface import ToolPluginInterface

# 导出公共接口
__all__ = [
    'ToolPluginInterface',
    'create_tool_plugin'
]


def create_tool_plugin(tool_type, plugin_id=None, config=None):
    """
    创建指定类型的工具插件

    此函数是创建工具插件的便捷工厂方法

    Args:
        tool_type: 工具类型，支持 "debug" 等
        plugin_id: 可选的插件ID，如果不提供则使用默认ID
        config: 可选的插件配置

    Returns:
        ToolPluginInterface: 创建的工具插件实例，或者None（如果创建失败）

    Raises:
        ValueError: 如果指定了不支持的工具类型
    """
    tool_type = tool_type.lower()

    # 根据类型导入并创建相应的工具插件
    if tool_type == "debug":
        from .debug_tool_plugin import create_plugin
        return create_plugin(plugin_id or "debug_tool", config)

    else:
        raise ValueError(f"不支持的工具类型: {tool_type}")
