# -*- coding: utf-8 -*-
"""
映射器插件包 - 包含各种位置映射器的插件实现

此包提供了多种不同的位置映射器插件，用于将检测到的人体坐标映射到虚拟房间坐标系统中。
目前包含:
- 高级映射器插件 (Advanced Mapper Plugin)

每个映射器插件都实现了标准的 MapperPluginInterface 接口，
可以无缝集成到插件系统中。
"""

from plugins.core.plugin_interface import MapperPluginInterface

# 导出公共接口
__all__ = [
    'MapperPluginInterface',
    'create_mapper_plugin'
]


def create_mapper_plugin(mapper_type, plugin_id=None, config=None):
    """
    创建指定类型的映射器插件

    此函数是创建映射器插件的便捷工厂方法

    Args:
        mapper_type: 映射器类型，支持 "advanced" 等
        plugin_id: 可选的插件ID，如果不提供则使用默认ID
        config: 可选的插件配置

    Returns:
        MapperPluginInterface: 创建的映射器插件实例，或者None（如果创建失败）

    Raises:
        ValueError: 如果指定了不支持的映射器类型
    """
    mapper_type = mapper_type.lower()

    # 根据类型导入并创建相应的映射器插件
    if mapper_type == "advanced":
        from .advanced_mapper_plugin import create_plugin
        return create_plugin(plugin_id or "advanced_mapper", config)

    else:
        raise ValueError(f"不支持的映射器类型: {mapper_type}")
