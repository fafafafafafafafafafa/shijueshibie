# -*- coding: utf-8 -*-
"""
增强型输入处理器 - 支持事件驱动的用户输入处理

此模块扩展了InputHandler类，添加了与UI事件系统的深度集成，
使用户输入可以通过事件系统分发，实现更灵活的处理方式。
"""
import cv2
import logging
from utils.logger_config import get_logger
from utils.ui_events import (
    UIEventTypes,
    UIEventPublisher,
    UIEventSubscriber,
    UIEventDrivenComponent
)

logger = get_logger("EnhancedInputHandler")


class EnhancedInputHandler(UIEventDrivenComponent):
    """
    增强型输入处理器 - 支持事件驱动的用户输入处理

    基于原始InputHandler，但添加了事件驱动的处理机制，
    将用户输入转换为事件并通过事件系统分发，允许多个组件响应同一输入。
    """

    def __init__(self, display_manager):
        """
        初始化增强型输入处理器

        Args:
            display_manager: 显示管理器实例
        """
        # 调用基类初始化
        super().__init__("input_handler")

        # 基本属性
        self.display_manager = display_manager
        self.handlers = {}  # 按键处理函数字典 {键码: 处理函数}
        self.exit_requested = False

        # 帮助信息
        self.help_text = {
            'q': '退出程序',
            'd': '切换调试信息显示',
            'r': '重新校准',
            'f': '切换FPS显示',
            's': '切换骨架显示',
            'b': '切换边界框显示',
            't': '切换轨迹显示',
            'a': '切换动作显示',
            'h': '显示此帮助',
            'm': '切换MediaPipe功能',
            'l': '切换机器学习模型',
            'w': '切换DTW功能',
            'p': '切换多线程处理',
            'y': '切换异步管道',
            'z': '显示缓存统计',
            'o': '生成缓存分析报告'
        }

        # 更新DisplayManager中的按键提示
        display_manager.key_hints = self.help_text

        # 功能切换回调
        self.feature_toggle_callbacks = {
            'm': None,  # MediaPipe
            'l': None,  # ML模型
            'w': None,  # DTW
            'p': None,  # 多线程
            'y': None  # 异步管道
        }

        # 注册内置处理函数
        self._register_default_handlers()

        logger.info("增强型输入处理器已初始化")

    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 目前没有需要订阅的事件，但保留此方法以便后续扩展
        pass

    def _register_default_handlers(self):
        """注册默认的按键处理函数"""
        # 注册缓存统计处理函数
        self.register_handler('z', self.show_cache_stats, '显示缓存统计')
        # 注册缓存分析报告处理函数
        self.register_handler('o', self.generate_cache_report,
                              '生成缓存分析报告')
        # 注册帮助函数
        self.register_handler('h', self.show_help, '显示帮助')

    def register_handler(self, key, callback, description=None):
        """注册按键处理函数

        Args:
            key: 按键字符
            callback: 回调函数
            description: 按键功能描述

        Returns:
            bool: 是否成功注册
        """
        try:
            self.handlers[ord(key)] = callback

            if description:
                self.help_text[key] = description
                # 同步更新DisplayManager中的提示
                if hasattr(self.display_manager, 'key_hints'):
                    self.display_manager.key_hints[key] = description

            logger.info(
                f"已注册按键 '{key}' 的处理函数: {description or '无描述'}")
            return True
        except Exception as e:
            logger.error(f"注册按键 '{key}' 的处理函数失败: {e}")
            return False

    def register_feature_toggle(self, feature_key, callback):
        """注册功能切换回调

        Args:
            feature_key: 功能对应的按键
            callback: 切换回调函数，应接受feature_name和新状态两个参数

        Returns:
            bool: 是否成功注册
        """
        if feature_key in self.feature_toggle_callbacks:
            self.feature_toggle_callbacks[feature_key] = callback
            logger.info(f"已注册功能 '{feature_key}' 的切换回调")
            return True

        logger.warning(f"尝试注册未知功能键 '{feature_key}' 的回调")
        return False

    def process_input(self):
        """处理用户输入

        Returns:
            bool: 是否请求退出程序
        """
        key = cv2.waitKey(1) & 0xFF

        if key == 255:  # 没有按键输入
            return False

        # 发布按键事件
        self.publisher.publish_user_interaction(
            UIEventTypes.KEY_PRESSED,
            "keyboard",
            key,
            char=chr(key) if 32 <= key <= 126 else None
        )

        if key == ord('q'):
            self.exit_requested = True
            logger.info("用户请求退出")
            # 发布通知
            self.publisher.publish_notification("用户请求退出，应用即将关闭",
                                                level="info")
            return True  # 退出

        # 检查是否有注册的处理函数
        if key in self.handlers:
            try:
                self.handlers[key]()
            except Exception as e:
                logger.error(f"处理按键 '{chr(key)}' 时出错: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # 发布错误通知
                self.publisher.publish_notification(
                    f"处理按键 '{chr(key)}' 时出错: {e}",
                    level="error"
                )

        # 处理基本显示选项
        self._handle_display_options(key)

        # 处理功能切换按键
        self._handle_feature_toggles(key)

        return False  # 继续运行

    def _handle_display_options(self, key):
        """处理显示选项相关的按键"""
        if key == ord('d'):
            # 切换调试显示模式
            if hasattr(self.display_manager, 'toggle_advanced_info'):
                self.display_manager.toggle_advanced_info()
                logger.info(
                    f"高级调试信息: {'开启' if self.display_manager.show_advanced_info else '关闭'}")

        elif key == ord('f'):
            # 切换FPS显示
            if hasattr(self.display_manager, 'toggle_display_option'):
                self.display_manager.toggle_display_option("show_fps")

        elif key == ord('s'):
            # 切换骨架显示
            if hasattr(self.display_manager, 'toggle_display_option'):
                self.display_manager.toggle_display_option("show_skeleton")

        elif key == ord('b'):
            # 切换边界框显示
            if hasattr(self.display_manager, 'toggle_display_option'):
                self.display_manager.toggle_display_option("show_bbox")

        elif key == ord('t'):
            # 切换轨迹显示
            if hasattr(self.display_manager, 'toggle_display_option'):
                self.display_manager.toggle_display_option("show_trail")

        elif key == ord('a'):
            # 切换动作显示
            if hasattr(self.display_manager, 'toggle_display_option'):
                self.display_manager.toggle_display_option("show_action")

    def _handle_feature_toggles(self, key):
        """处理功能切换相关的按键"""
        feature_keys = {'m', 'l', 'w', 'p', 'y'}

        if chr(key) in feature_keys:
            feature_key = chr(key)

            if feature_key in self.feature_toggle_callbacks and \
                    self.feature_toggle_callbacks[feature_key]:
                feature_name = {
                    'm': 'mediapipe',
                    'l': 'ml_model',
                    'w': 'dtw',
                    'p': 'threading',
                    'y': 'async'
                }.get(feature_key)

                # 获取当前状态并切换
                current_state = self.display_manager.feature_states.get(
                    feature_name, False)
                new_state = not current_state

                # 调用回调函数
                try:
                    success = self.feature_toggle_callbacks[feature_key](
                        feature_name, new_state)

                    if success:
                        # 更新显示状态
                        self.display_manager.update_feature_state(feature_name,
                                                                  new_state)
                        logger.info(
                            f"功能 '{feature_name}' 已{'启用' if new_state else '禁用'}")

                        # 发布功能切换事件
                        self.publisher.publish_ui_state_change(
                            UIEventTypes.FEATURE_TOGGLED,
                            feature_name,
                            current_state,
                            new_state
                        )
                    else:
                        logger.warning(f"功能 '{feature_name}' 切换失败")
                        # 发布失败通知
                        self.publisher.publish_notification(
                            f"功能 '{feature_name}' 切换失败",
                            level="warning"
                        )
                except Exception as e:
                    logger.error(f"切换功能 '{feature_name}' 时出错: {e}")
                    # 发布错误通知
                    self.publisher.publish_notification(
                        f"切换功能 '{feature_name}' 时出错: {e}",
                        level="error"
                    )

    def is_exit_requested(self):
        """检查是否请求退出

        Returns:
            bool: 是否请求退出
        """
        return self.exit_requested

    def show_help(self):
        """显示帮助信息"""
        # 生成帮助通知
        self.publisher.publish_notification(
            "按键帮助已显示，查看窗口",
            level="info"
        )

        # 在控制台显示帮助
        print("\n=== 键盘操作帮助 ===")
        for key, desc in sorted(self.help_text.items()):
            print(f"{key}: {desc}")
        print("==================\n")

        # 创建一个单独的窗口显示所有快捷键信息
        help_window = self._create_help_window()
        cv2.imshow("键盘快捷键帮助", help_window)
        cv2.waitKey(100)  # 短暂等待确保窗口显示

    def _create_help_window(self, width=500, height=600):
        """创建一个显示所有快捷键的窗口

        Args:
            width: 窗口宽度
            height: 窗口高度

        Returns:
            ndarray: 帮助窗口图像
        """
        import numpy as np

        # 创建一个白色背景
        help_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 添加标题
        cv2.putText(help_img, "键盘快捷键帮助",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # 添加分隔线
        cv2.line(help_img, (20, 60), (width - 20, 60), (0, 0, 0), 1)

        # 添加功能区域标题
        cv2.putText(help_img, "基本功能:",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)

        # 添加高级功能标题
        advanced_y = 350  # 高级功能的开始Y坐标
        cv2.putText(help_img, "高级功能开关:",
                    (20, advanced_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (150, 0, 0), 2)
        cv2.line(help_img, (20, advanced_y - 10),
                 (width - 20, advanced_y - 10),
                 (100, 100, 100), 1)

        # 添加每个快捷键的说明
        basic_keys = ['q', 'd', 'r', 'f', 's', 'b', 't', 'a', 'h', 'z', 'o']
        advanced_keys = ['m', 'l', 'w', 'p', 'y']

        # 基本功能键
        y_pos = 120
        for key in basic_keys:
            desc = self.help_text.get(key, "")
            # 绘制按键背景
            cv2.rectangle(help_img, (30, y_pos - 25), (60, y_pos + 5),
                          (200, 200, 200), -1)
            cv2.rectangle(help_img, (30, y_pos - 25), (60, y_pos + 5),
                          (100, 100, 100), 1)

            # 绘制按键和描述
            cv2.putText(help_img, key, (40, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)
            cv2.putText(help_img, ": " + desc, (70, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_pos += 40

        # 高级功能键
        y_pos = advanced_y
        for key in advanced_keys:
            desc = self.help_text.get(key, "")
            # 绘制按键背景
            cv2.rectangle(help_img, (30, y_pos - 25), (60, y_pos + 5),
                          (200, 200, 200), -1)
            cv2.rectangle(help_img, (30, y_pos - 25), (60, y_pos + 5),
                          (100, 100, 100), 1)

            # 绘制按键和描述
            cv2.putText(help_img, key, (40, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 2)
            cv2.putText(help_img, ": " + desc, (70, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_pos += 40

        return help_img


