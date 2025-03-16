# -*- coding: utf-8 -*-
"""
输入处理器模块 - 处理用户输入，支持键盘命令和回调函数

增强版: 支持事件驱动的输入处理，同时保持原有接口不变。
"""
import time  # 添加这一行
import cv2
import logging
import numpy as np
from utils.logger_config import get_logger
from utils.ui_events import (
    UIEventTypes,
    UIEventPublisher,
    UIEventSubscriber
)
from core.component_interface import BaseLifecycleComponent
from core.event_bus import get_event_bus
from core.component_lifecycle import LifecycleState

logger = get_logger("InputHandler")


class InputHandler(BaseLifecycleComponent):
    """
    输入处理器 - 处理用户输入，支持键盘命令和回调函数

    内部采用事件驱动机制，同时保持与原始接口的兼容性。
    """

    def __init__(self, display_manager,component_id="input_handler", component_type="UI"):
        """
        初始化输入处理器

        Args:
            display_manager: 显示管理器实例
        """
        # 初始化基础生命周期组件
        super().__init__(component_id, component_type)

        # 基本属性
        self.display_manager = display_manager
        self.handlers = {}  # 按键处理函数字典 {键码: 处理函数}
        self.exit_requested = False
        self.logger = logger

        # 事件系统支持
        self.events = None  # 传统API兼容
        self.publisher = UIEventPublisher.get_instance("input_handler")
        self.subscriber = UIEventSubscriber.get_instance("input_handler")

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
        if hasattr(display_manager, 'key_hints'):
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

        # 获取事件总线
        self.event_bus = get_event_bus()

        logger.info("输入处理器已初始化")

    def initialize(self) -> bool:
        """初始化组件"""
        try:
            # 确保组件已注册
            if self.get_state() == LifecycleState.UNREGISTERED:
                self._lifecycle_manager.transition_to(LifecycleState.REGISTERED)
                self.logger.info(f"{self._component_type} 已注册")

            # 执行初始化
            result = self._do_initialize()

            if result:
                # 如果初始化成功且当前状态是 REGISTERED，转换到 INITIALIZED
                if self.get_state() == LifecycleState.REGISTERED:
                    self._lifecycle_manager.transition_to(
                        LifecycleState.INITIALIZING)
                    self._lifecycle_manager.transition_to(
                        LifecycleState.INITIALIZED)
                    self.logger.info(f"{self._component_type} 已初始化")

            return result
        except Exception as e:
            self.logger.error(f"初始化 {self._component_type} 时出错: {e}")
            # 设置错误状态
            if self.get_state() != LifecycleState.UNREGISTERED:
                self._lifecycle_manager.transition_to(LifecycleState.ERROR)
            return False

    def set_display_manager(self, display_manager):
        """设置显示管理器"""
        self.display_manager = display_manager
        # 更新DisplayManager中的按键提示
        if hasattr(display_manager, 'key_hints'):
            display_manager.key_hints = self.help_text

    # 实现生命周期方法
    def _do_initialize(self) -> bool:
        """执行初始化逻辑"""
        try:
            # 注册内置处理函数
            self._register_default_handlers()

            # 设置事件处理器
            self._setup_event_handlers()

            logger.info("输入处理器已初始化")
            return True
        except Exception as e:
            logger.error(f"初始化输入处理器时出错: {e}")
            return False

    def _do_start(self) -> bool:
        """执行启动逻辑"""
        try:

            if self.publisher:
                self.publisher.publish_notification(
                    "输入处理器已启动",
                    level="info"
                )

            logger.info("输入处理器已启动")
            return True
        except Exception as e:
            logger.error(f"启动输入处理器时出错: {e}")
            return False

    def _do_stop(self) -> bool:
        """执行停止逻辑"""
        try:
            # 取消所有事件订阅
            if hasattr(self.subscriber, 'unsubscribe_all'):
                self.subscriber.unsubscribe_all()

            logger.info("输入处理器已停止")
            return True
        except Exception as e:
            logger.error(f"停止输入处理器时出错: {e}")
            return False

    def _do_destroy(self) -> bool:
        """执行销毁逻辑"""
        return self.cleanup()

    def _register_default_handlers(self):
        """注册默认的按键处理函数"""
        # 注册缓存统计处理函数
        self.register_handler('z', self.show_cache_stats, '显示缓存统计')
        # 注册缓存分析报告处理函数
        self.register_handler('o', self.generate_cache_report,
                              '生成缓存分析报告')
        # 注册帮助函数
        self.register_handler('h', self.show_help, '显示帮助')

    def set_event_system(self, event_system):
        """
        设置事件系统引用

        为保持API兼容性而保留

        Args:
            event_system: 事件系统实例
        """
        self.events = event_system

        # 如果有发布器和订阅器，更新它们的事件系统
        # 重新获取发布器和订阅器实例
        self.publisher = UIEventPublisher.get_instance("input_handler")
        self.subscriber = UIEventSubscriber.get_instance("input_handler")

        # 设置事件处理器
        self._setup_event_handlers()

        self.logger.info("已设置事件系统引用")

    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 只有在有订阅器时才设置
        if not hasattr(self.subscriber, 'subscribe'):
            return

        # 订阅UI事件
        self.subscriber.subscribe(
            UIEventTypes.KEY_PRESSED,
            self._handle_key_event
        )

        # 订阅UI状态变更事件
        self.subscriber.subscribe(
            UIEventTypes.OPTION_TOGGLED,
            self._handle_option_toggle
        )

        # 订阅性能警告事件
        self.subscriber.subscribe(
            UIEventTypes.PERFORMANCE_WARNING,
            self._handle_performance_warning
        )

        self.logger.info("已设置输入处理器事件处理器")

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
                # 标记帮助内容已变化，需要重新生成帮助窗口
                self._help_content_changed = True
                if hasattr(self.display_manager, 'key_hints'):
                    self.display_manager.key_hints[key] = description

            self.logger.info(
                f"已注册按键 '{key}' 的处理函数: {description or '无描述'}")
            return True
        except Exception as e:
            self.logger.error(f"注册按键 '{key}' 的处理函数失败: {e}")
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
            self.logger.info(f"已注册功能 '{feature_key}' 的切换回调")
            return True

        self.logger.warning(f"尝试注册未知功能键 '{feature_key}' 的回调")
        return False

    def _is_continuous_key(self, key):
        """判断是否是连续触发的按键，如方向键等"""
        # 定义需要节流的连续按键列表
        continuous_keys = [
            0x25, 0x26, 0x27, 0x28,  # 方向键 左上右下
            ord('w'), ord('a'), ord('s'), ord('d'),  # WASD控制键
            # 添加其他需要节流的按键
        ]
        return key in continuous_keys

    def process_input(self):
        """处理用户输入

        Returns:
            bool: 是否请求退出程序
        """
        key = cv2.waitKey(1) & 0xFF

        if key == 255:  # 没有按键输入
            return False

        # 添加节流逻辑: 对于高频事件进行节流处理
        current_time = time.time()

        # 每种事件类型可以设置不同的节流间隔
        throttle_intervals = {
            'mouse_move': 0.05,  # 鼠标移动每50ms处理一次
            'key_continuous': 0.1,  # 持续按键(如方向键)每100ms处理一次
        }

        # 存储上次处理特定类型事件的时间
        if not hasattr(self, '_last_event_times'):
            self._last_event_times = {}

        # 判断是否是需要节流的高频事件
        event_type = 'key_continuous' if self._is_continuous_key(
            key) else None

        # 如果是需要节流的事件，检查时间间隔
        if event_type and event_type in throttle_intervals:
            last_time = self._last_event_times.get(event_type, 0)
            if current_time - last_time < throttle_intervals[event_type]:
                # 未达到节流间隔，跳过处理
                return False
            # 更新最后处理时间
            self._last_event_times[event_type] = current_time

        # 发布按键事件
        if self.publisher:
            self.publisher.publish_user_interaction(
                UIEventTypes.KEY_PRESSED,
                "keyboard",
                key,
                char=chr(key) if 32 <= key <= 126 else None
            )

        if key == ord('q'):
            self.exit_requested = True
            self.logger.info("用户请求退出")
            # 发布通知
            if self.publisher:
                self.publisher.publish_notification(
                    "用户请求退出，应用即将关闭",
                    level="info"
                )
            return True  # 退出

        # 检查是否有注册的处理函数
        if key in self.handlers:
            try:
                self.handlers[key]()
            except Exception as e:
                self.logger.error(f"处理按键 '{chr(key)}' 时出错: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                # 发布错误通知
                if self.publisher:
                    self.publisher.publish_notification(
                        f"处理按键 '{chr(key)}' 时出错: {e}",
                        level="error"
                    )

        # 处理基本显示选项
        self._handle_display_options(key)

        # 处理功能切换按键
        self._handle_feature_toggles(key)

        return False  # 继续运行

    def _handle_key_event(self, data):
        """处理按键事件

        Args:
            data: 事件数据
        """
        key = data.get('value')
        if key is None:
            return

        # 仅处理特定事件源的键盘事件
        if data.get('interaction_type') != 'keyboard':
            return

        # 处理标准按键
        if key == ord('q'):
            self.exit_requested = True
            self.logger.info("通过事件请求退出")
            return

        # 检查是否有注册的处理函数 (不执行，避免重复处理)
        if key in self.handlers:
            # 记录事件接收
            self.logger.debug(f"接收到按键事件: '{chr(key)}'")

    def _handle_option_toggle(self, data):
        """处理选项切换事件

        Args:
            data: 事件数据
        """
        option = data.get('state_name')
        new_value = data.get('new_value')

        self.logger.info(f"选项 '{option}' 已更改为: {new_value}")

    def _handle_performance_warning(self, data):
        """处理性能警告事件

        Args:
            data: 事件数据
        """
        warning_type = data.get('warning_type')
        if warning_type == 'low_fps':
            fps = data.get('fps', 0)
            self.logger.warning(f"性能警告: FPS={fps:.1f}")

    def _handle_display_options(self, key):
        """处理显示选项相关的按键"""
        if key == ord('d'):
            # 切换调试显示模式
            if hasattr(self.display_manager, 'toggle_advanced_info'):
                self.display_manager.toggle_advanced_info()
                self.logger.info(
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
        feature_keys = {'m', 'l', 'w', 'p'}

        if chr(key) in feature_keys:
            feature_key = chr(key)

            if feature_key in self.feature_toggle_callbacks and \
                    self.feature_toggle_callbacks[feature_key]:
                feature_name = {
                    'm': 'mediapipe',
                    'l': 'ml_model',
                    'w': 'dtw',
                    'p': 'threading'
                }.get(feature_key)

                # 添加空值检查
                if self.display_manager is None:
                    self.logger.error("display_manager 未初始化，无法切换功能")
                    if self.publisher:
                        self.publisher.publish_notification(
                            "display_manager 未初始化，无法切换功能",
                            level="error"
                        )
                    return

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
                        self.logger.info(
                            f"功能 '{feature_name}' 已{'启用' if new_state else '禁用'}")

                        # 发布功能切换事件
                        if self.publisher:
                            self.publisher.publish_ui_state_change(
                                UIEventTypes.FEATURE_TOGGLED,
                                feature_name,
                                current_state,
                                new_state
                            )
                    else:
                        self.logger.warning(f"功能 '{feature_name}' 切换失败")
                        # 发布失败通知
                        if self.publisher:
                            self.publisher.publish_notification(
                                f"功能 '{feature_name}' 切换失败",
                                level="warning"
                            )
                except Exception as e:
                    self.logger.error(f"切换功能 '{feature_name}' 时出错: {e}")
                    # 发布错误通知
                    if self.publisher:
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
        if self.publisher:
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

    def show_help(self):
        """显示帮助信息"""
        # 生成帮助通知
        if self.publisher:
            self.publisher.publish_notification(
                "按键帮助已显示，查看窗口",
                level="info"
            )

        # 在控制台显示帮助
        print("\n=== 键盘操作帮助 ===")
        for key, desc in sorted(self.help_text.items()):
            print(f"{key}: {desc}")
        print("==================\n")

        # 使用缓存的帮助窗口图像
        if not hasattr(self,
                       '_cached_help_window') or self._cached_help_window is None:
            self._cached_help_window = self._create_help_window()
            # 缓存失效条件: 如果帮助内容变化，设置一个标志
            self._help_content_changed = False
        elif self._help_content_changed:
            # 内容变化时重新生成
            self._cached_help_window = self._create_help_window()
            self._help_content_changed = False

        # 显示缓存的窗口
        cv2.imshow("键盘快捷键帮助", self._cached_help_window)
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

    def show_cache_stats(self):
        """显示缓存统计信息"""
        try:
            from utils.cache_monitor import get_monitor

            monitor = get_monitor()

            # 确保监控线程运行
            if not monitor.monitoring:
                monitor.start_monitoring()

            # 立即持久化当前统计
            for name in monitor.monitored_caches:
                monitor._persist_stats(name)

            # 打印统计
            stats = monitor.print_stats()

            # 发布通知
            if self.publisher:
                self.publisher.publish_notification(
                    "缓存统计数据已更新，查看控制台输出",
                    level="info"
                )

            # 触发UI更新
            if hasattr(self.display_manager, '_update_ui'):
                self.display_manager._update_ui()

            # 提醒用户
            print("\n缓存统计数据已保存到 logs/cache_stats/ 目录")

            return stats
        except Exception as e:
            self.logger.error(f"显示缓存统计信息时出错: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

            # 发布错误通知
            if self.publisher:
                self.publisher.publish_notification(
                    f"显示缓存统计信息时出错: {e}",
                    level="error"
                )

            return None

    def generate_cache_report(self):
        """生成缓存分析报告"""
        try:
            # 确保缓存统计目录存在
            import os
            stats_dir = "logs/cache_stats"
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir, exist_ok=True)
                print(f"创建缓存统计目录: {stats_dir}")

            from utils.cache_analyzer import analyze_caches
            report_file = analyze_caches(stats_dir=stats_dir)

            if report_file and os.path.exists(report_file):
                print(f"\n缓存分析报告已生成在: {report_file}")

                # 发布通知
                if self.publisher:
                    self.publisher.publish_notification(
                        f"缓存分析报告已生成: {report_file}",
                        level="info"
                    )

                # 显示报告预览
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        preview = f.read(500)  # 读取前500个字符
                        print("\n报告预览:")
                        print("-" * 40)
                        print(preview + "...")
                        print("-" * 40)
                except Exception as e:
                    print(f"无法显示报告预览: {e}")

                return report_file
            else:
                print("无法生成缓存分析报告，请确保缓存系统正常运行")

                # 发布警告通知
                if self.publisher:
                    self.publisher.publish_notification(
                        "无法生成缓存分析报告，请确保缓存系统正常运行",
                        level="warning"
                    )

                return None

        except Exception as e:
            self.logger.error(f"生成缓存分析报告时出错: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            print(f"生成缓存分析报告时出错: {e}")

            # 发布错误通知
            if self.publisher:
                self.publisher.publish_notification(
                    f"生成缓存分析报告时出错: {e}",
                    level="error"
                )

            return None

    def cleanup(self):
        """
        清理资源

        在应用程序退出前调用，取消事件订阅

        Returns:
            bool: 是否成功清理
        """
        try:
            # 取消事件订阅
            if hasattr(self.subscriber, 'unsubscribe_all'):
                self.subscriber.unsubscribe_all()

            # 清理发布器资源
            if hasattr(self.publisher, 'cleanup'):
                self.publisher.cleanup()

            self.logger.info("输入处理器资源已清理")
            return True
        except Exception as e:
            self.logger.error(f"清理输入处理器资源时出错: {e}")
            return False


