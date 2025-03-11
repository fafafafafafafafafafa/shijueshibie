# plugin_tracker_app.py
import cv2
import time
import logging
import os
import numpy as np
from typing import Dict, List, Any, Optional


# 导入原始应用组件

from utils.logger_config import setup_logger, init_root_logger, setup_utf8_console
# 修改为
from core.tracker_app import TrackerApp
from ui.display_manager import DisplayManager
from ui.input_handler import InputHandler

logger = setup_logger("PluginTrackerApp")


class PluginTrackerApp(TrackerApp):
    """支持插件系统的追踪应用程序类"""

    def __init__(self, default_detector=None, default_action_recognizer=None,
                 position_mapper=None, visualizer=None):
        """
        初始化支持插件的追踪应用程序

        Args:
            default_detector: 默认检测器（如果没有检测器插件）
            default_action_recognizer: 默认动作识别器（如果没有识别器插件）
            position_mapper: 位置映射器
            visualizer: 可视化器
        """
        # 初始化插件系统
        self.detector_plugins = {}
        self.recognizer_plugins = {}
        self.active_detector_id = None
        self.active_recognizer_id = None

        detector = self._select_detector(default_detector)
        action_recognizer = self._select_recognizer(default_action_recognizer)

        print(f"使用的检测器类型: {type(detector).__name__}")
        if hasattr(detector, 'detect_pose'):
            print("检测器具有detect_pose方法")
        else:
            print("警告：检测器缺少detect_pose方法")
        print(f"使用的识别器类型: {type(action_recognizer).__name__}")

        # 尝试加载插件
        self._init_plugin_system()

        # 选择要使用的组件（插件或默认）
        detector = self._select_detector(default_detector)
        action_recognizer = self._select_recognizer(default_action_recognizer)

        # 调用父类初始化
        super().__init__(detector, action_recognizer, position_mapper,
                         visualizer)

        # 添加插件管理菜单项
        self.input_handler.register_handler('p', self.show_plugin_menu,
                                            '插件菜单')

        logger.info("插件追踪应用初始化完成")

    def _init_plugin_system(self):
        """初始化插件系统"""
        try:
            # 确保插件目录存在
            plugin_dirs = ['plugins', 'plugin_modules']
            for plugin_dir in plugin_dirs:
                if not os.path.exists(plugin_dir):
                    os.makedirs(plugin_dir)
                    with open(os.path.join(plugin_dir, '__init__.py'),
                              'w') as f:
                        pass  # 创建空的__init__.py文件

            # 尝试导入并加载检测器插件
            try:
                from plugin_modules.yolo_detector import YOLOv8Detector
                detector = YOLOv8Detector()
                self.detector_plugins['yolov8_detector'] = detector
                self.active_detector_id = 'yolov8_detector'
                logger.info("已加载YOLOv8检测器插件")
            except ImportError:
                logger.warning("未找到YOLOv8检测器插件")

            # 尝试导入并加载识别器插件
            try:
                from plugin_modules.rule_recognizer import RuleActionRecognizer
                recognizer = RuleActionRecognizer()
                self.recognizer_plugins['rule_action_recognizer'] = recognizer
                self.active_recognizer_id = 'rule_action_recognizer'
                logger.info("已加载规则动作识别器插件")
            except ImportError:
                logger.warning("未找到规则动作识别器插件")

            #在加载插件后，初始化它们
            if 'yolov8_detector' in self.detector_plugins:
                self.detector_plugins['yolov8_detector'].initialize(
                    self.config or {})
            if 'rule_action_recognizer' in self.recognizer_plugins:
                self.recognizer_plugins['rule_action_recognizer'].initialize(
                    self.config or {})


        except Exception as e:
            logger.error(f"初始化插件系统时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())



    def _select_recognizer(self, default_recognizer):
        """选择使用的识别器"""
        if self.active_recognizer_id and self.active_recognizer_id in self.recognizer_plugins:
            logger.info(f"使用识别器插件: {self.active_recognizer_id}")
            return self.recognizer_plugins[self.active_recognizer_id]
        else:
            logger.info("使用默认识别器")
            return default_recognizer

    def show_plugin_menu(self):
        """显示插件管理菜单"""
        # 创建一个窗口显示插件信息
        menu_img = self._create_plugin_menu_window()
        cv2.imshow("插件管理", menu_img)

        # 等待按键，处理插件选择
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27 or key == ord('q'):  # ESC或Q退出
                break
            elif key == ord('r'):  # R重载插件
                self._reload_plugins()
                menu_img = self._create_plugin_menu_window()
                cv2.imshow("插件管理", menu_img)
            elif key >= ord('1') and key <= ord('9'):  # 数字1-9选择检测器
                index = key - ord('1')
                self._select_detector_by_index(index)
                menu_img = self._create_plugin_menu_window()
                cv2.imshow("插件管理", menu_img)

        cv2.destroyWindow("插件管理")

    def _create_plugin_menu_window(self, width=600, height=500):
        """创建插件菜单窗口"""
        # 创建一个白色背景
        menu_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 添加标题
        cv2.putText(menu_img, "插件管理",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # 添加分隔线
        cv2.line(menu_img, (20, 60), (width - 20, 60), (0, 0, 0), 1)

        # 显示检测器插件
        y_pos = 100
        cv2.putText(menu_img, "检测器插件:",
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)
        y_pos += 30

        if not self.detector_plugins:
            cv2.putText(menu_img, "没有检测器插件",
                        (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (100, 100, 100), 1)
            y_pos += 30
        else:
            for i, (plugin_id, plugin) in enumerate(
                    self.detector_plugins.items()):
                text = f"{i + 1}. {plugin_id}"
                color = (
                0, 150, 0) if plugin_id == self.active_detector_id else (
                0, 0, 0)
                cv2.putText(menu_img, text,
                            (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                            1)
                y_pos += 25

        # 显示识别器插件
        y_pos += 20
        cv2.putText(menu_img, "识别器插件:",
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 2)
        y_pos += 30

        if not self.recognizer_plugins:
            cv2.putText(menu_img, "没有识别器插件",
                        (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (100, 100, 100), 1)
            y_pos += 30
        else:
            for i, (plugin_id, plugin) in enumerate(
                    self.recognizer_plugins.items()):
                text = f"{i + 1}. {plugin_id}"
                color = (
                0, 150, 0) if plugin_id == self.active_recognizer_id else (
                0, 0, 0)
                cv2.putText(menu_img, text,
                            (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                            1)
                y_pos += 25

        # 添加帮助说明
        help_y = height - 80
        cv2.putText(menu_img, "插件控制:",
                    (20, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        help_y += 25
        cv2.putText(menu_img, "1-9: 选择检测器插件  |  R: 重新加载插件",
                    (40, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        help_y += 25
        cv2.putText(menu_img, "ESC/Q: 关闭菜单",
                    (40, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return menu_img

    def _select_detector_by_index(self, index):
        """通过索引选择检测器"""
        if not self.detector_plugins or index >= len(self.detector_plugins):
            logger.warning(f"无效的检测器索引: {index}")
            return False

        try:
            # 获取索引对应的插件ID
            plugin_id = list(self.detector_plugins.keys())[index]

            # 更新活动检测器
            self.active_detector_id = plugin_id
            self.detector = self.detector_plugins[plugin_id]

            logger.info(f"已切换到检测器插件: {plugin_id}")
            return True
        except Exception as e:
            logger.error(f"切换检测器插件时出错: {e}")
            return False

    def _reload_plugins(self):
        """重新加载插件"""
        try:
            # 保存当前活动插件ID
            active_detector_id = self.active_detector_id
            active_recognizer_id = self.active_recognizer_id

            # 清空插件记录
            self.detector_plugins = {}
            self.recognizer_plugins = {}

            # 重新初始化插件系统
            self._init_plugin_system()

            # 尝试恢复之前的活动插件
            if active_detector_id in self.detector_plugins:
                self.active_detector_id = active_detector_id
                self.detector = self.detector_plugins[active_detector_id]

            if active_recognizer_id in self.recognizer_plugins:
                self.active_recognizer_id = active_recognizer_id
                self.action_recognizer = self.recognizer_plugins[
                    active_recognizer_id]

            logger.info("插件重新加载完成")
            return True
        except Exception as e:
            logger.error(f"重新加载插件时出错: {e}")
            return False

    def cleanup(self):
        """清理资源（重写父类方法）"""
        try:
            # 清理插件资源
            for plugin_id, plugin in self.detector_plugins.items():
                try:
                    if hasattr(plugin, 'release'):
                        plugin.release()
                except Exception as e:
                    logger.error(f"释放检测器插件 {plugin_id} 资源时出错: {e}")

            for plugin_id, plugin in self.recognizer_plugins.items():
                try:
                    if hasattr(plugin, 'release'):
                        plugin.release()
                except Exception as e:
                    logger.error(f"释放识别器插件 {plugin_id} 资源时出错: {e}")

            # 调用父类清理方法
            super().cleanup()
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")

    def _select_detector(self, default_detector):
        """选择使用的检测器"""
        if self.active_detector_id and self.active_detector_id in self.detector_plugins:
            logger.info(f"使用检测器插件: {self.active_detector_id}")
            plugin = self.detector_plugins[self.active_detector_id]

            # 如果插件没有detect_pose方法，创建一个适配器
            if not hasattr(plugin, 'detect_pose'):
                original_detect = plugin.detect
                plugin.detect_pose = original_detect

            return plugin
        else:
            logger.info("使用默认检测器")
            return default_detector
