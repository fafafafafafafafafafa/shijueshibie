# -*- coding: utf-8 -*-
"""
识别器插件包 - 包含各种动作识别器的插件实现

此包提供了多种不同的动作识别器插件，包括:
- 机器学习(ML)识别器插件
- 动态时间规整(DTW)识别器插件

每个识别器插件都实现了标准的 RecognizerPluginInterface 接口，
可以无缝集成到插件系统中。
"""

from plugins.core.plugin_interface import RecognizerPluginInterface

# 导出公共接口
__all__ = [
    'RecognizerPluginInterface',
    'create_recognizer_plugin'
]


def create_recognizer_plugin(recognizer_type, plugin_id=None, config=None):
    """
    创建指定类型的识别器插件

    此函数是创建识别器插件的便捷工厂方法

    Args:
        recognizer_type: 识别器类型，支持 "ml" 和 "dtw"
        plugin_id: 可选的插件ID，如果不提供则使用默认ID
        config: 可选的插件配置

    Returns:
        RecognizerPluginInterface: 创建的识别器插件实例，或者None（如果创建失败）

    Raises:
        ValueError: 如果指定了不支持的识别器类型
    """
    recognizer_type = recognizer_type.lower()

    # 根据类型导入并创建相应的识别器插件
    if recognizer_type == "ml":
        from .ml_recognizer_plugin import create_plugin
        return create_plugin(plugin_id or "ml_recognizer", config)

    elif recognizer_type == "dtw":
        from .dtw_recognizer_plugin import create_plugin
        return create_plugin(plugin_id or "dtw_recognizer", config)

    else:
        raise ValueError(f"不支持的识别器类型: {recognizer_type}")
