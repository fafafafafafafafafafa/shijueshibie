# -*- coding: utf-8 -*-
"""
检测器插件包 - 包含各种人体检测器的插件实现

此包提供了多种不同的人体检测器插件，包括:
- YOLO 检测器插件
- MediaPipe 检测器插件

每个检测器插件都实现了标准的 DetectorPluginInterface 接口，
可以无缝集成到插件系统中。
"""

from plugins.core.plugin_interface import DetectorPluginInterface

# 导出公共接口
__all__ = [
    'DetectorPluginInterface',
    'create_detector_plugin'
]


def create_detector_plugin(detector_type, plugin_id=None, config=None):
    """
    创建指定类型的检测器插件

    此函数是创建检测器插件的便捷工厂方法

    Args:
        detector_type: 检测器类型，支持 "yolo" 和 "mediapipe"
        plugin_id: 可选的插件ID，如果不提供则使用默认ID
        config: 可选的插件配置

    Returns:
        DetectorPluginInterface: 创建的检测器插件实例，或者None（如果创建失败）

    Raises:
        ValueError: 如果指定了不支持的检测器类型
    """
    detector_type = detector_type.lower()

    # 根据类型导入并创建相应的检测器插件
    if detector_type == "yolo":
        from .yolo_detector_plugin import create_plugin
        return create_plugin(plugin_id or "yolo_detector", config)

    elif detector_type == "mediapipe":
        from .mediapipe_detector_plugin import create_plugin
        return create_plugin(plugin_id or "mediapipe_detector", config)

    else:
        raise ValueError(f"不支持的检测器类型: {detector_type}")
