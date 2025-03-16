# -*- coding: utf-8 -*-
"""
插件开发模板包 - 提供插件开发的基础模板

此包为插件开发者提供了模板代码和示例，可用于快速启动
新插件的开发。包含标准接口实现和最佳实践。
"""

# 导入核心插件接口
from plugins.core.plugin_interface import PluginInterface

# 导出模板类
from .plugin_template import TemplatePlugin

# 导出公共接口
__all__ = [
    'PluginInterface',
    'TemplatePlugin'
]
