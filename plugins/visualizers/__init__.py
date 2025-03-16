# -*- coding: utf-8 -*-
"""
可视化器插件包 - 包含各种可视化插件的实现

此包提供了多种不同的可视化器插件，包括:
- 3D可视化器插件

每个可视化器插件都实现了标准的 VisualizerPluginInterface 接口，
可以无缝集成到插件系统中。
"""

from plugins.core.plugin_interface import VisualizerPluginInterface

# 导出公共接口
__all__ = [
    'VisualizerPluginInterface',
    'create_visualizer_plugin'
]


def create_visualizer_plugin(visualizer_type, plugin_id=None, config=None):
    """
    创建指定类型的可视化器插件

    此函数是创建可视化器插件的便捷工厂方法

    Args:
        visualizer_type: 可视化器类型，支持 "3d" 等
        plugin_id: 可选的插件ID，如果不提供则使用默认ID
        config: 可选的插件配置

    Returns:
        VisualizerPluginInterface: 创建的可视化器插件实例，或者None（如果创建失败）

    Raises:
        ValueError: 如果指定了不支持的可视化器类型
    """
    visualizer_type = visualizer_type.lower()

    # 根据类型导入并创建相应的可视化器插件
    if visualizer_type == "3d":
        from .3_d_visualizer_plugin import create_plugin
        return create_plugin(plugin_id or "3d_visualizer", config)

    else:
        raise ValueError(f"不支持的可视化器类型: {visualizer_type}")
