
# -*- coding: utf-8 -*-
import cv2
import logging
import numpy as np
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class InputHandler:
    """处理用户输入，支持键盘命令和回调函数 - 兼容SimplifiedPersonDetector"""

    def __init__(self, display_manager):
        self.display_manager = display_manager
        self.handlers = {}
        self.exit_requested = False
        self.logger = logger
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

        # 事件系统引用
        self.events = None

        # 注册缓存统计处理函数
        self.register_handler('z', self.show_cache_stats, '显示缓存统计')
        # 注册缓存分析报告处理函数
        self.register_handler('o', self.generate_cache_report,
                              '生成缓存分析报告')

    def set_event_system(self, event_system):
        """设置事件系统"""
        self.events = event_system

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
            else:
                print("无法生成缓存分析报告，请确保缓存系统正常运行")

        except Exception as e:
            print(f"生成缓存分析报告时出错: {e}")
            import traceback
            print(traceback.format_exc())  # 打印完整的错误堆栈

    def register_handler(self, key, callback, description=None):
        """注册按键处理函数

        Args:
            key: 按键字符
            callback: 回调函数
            description: 按键功能描述
        """
        self.handlers[ord(key)] = callback
        if description:
            self.help_text[key] = description
            # 同步更新DisplayManager中的提示
            self.display_manager.key_hints[key] = description

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
            return True
        return False

    def show_help(self):
        """显示帮助信息"""
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
        cv2.line(help_img, (20, advanced_y - 10), (width - 20, advanced_y - 10),
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

    def process_input(self):
        """处理用户输入

        Returns:
            bool: 是否请求退出程序
        """
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.exit_requested = True
            self.logger.info("用户请求退出")
            return True  # 退出

        if key == ord('h'):
            self.show_help()
            return False

        if key in self.handlers:
            try:
                self.handlers[key]()
            except Exception as e:
                self.logger.error(f"处理按键 '{chr(key)}' 时出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # 处理基本显示选项
        if key == ord('d'):
            # 切换调试显示模式
            self.display_manager.toggle_advanced_info()
            print(
                f"高级调试信息: {'开启' if self.display_manager.show_advanced_info else '关闭'}")
        elif key == ord('f'):
            # 切换FPS显示
            self.display_manager.toggle_display_option("show_fps")
        elif key == ord('s'):
            # 切换骨架显示
            self.display_manager.toggle_display_option("show_skeleton")
        elif key == ord('b'):
            # 切换边界框显示
            self.display_manager.toggle_display_option("show_bbox")
        elif key == ord('t'):
            # 切换轨迹显示
            self.display_manager.toggle_display_option("show_trail")
        elif key == ord('a'):
            # 切换动作显示
            self.display_manager.toggle_display_option("show_action")

        # 处理功能切换按键
        elif key in [ord('m'), ord('l'), ord('w'), ord('p'), ord('y')]:
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
                success = self.feature_toggle_callbacks[feature_key](
                    feature_name, new_state)
                if success:
                    # 更新显示状态
                    self.display_manager.update_feature_state(feature_name,
                                                              new_state)
                    print(
                        f"功能 '{feature_name}' 已{'启用' if new_state else '禁用'}")
                else:
                    print(f"功能 '{feature_name}' 切换失败")

        return False  # 继续运行

    def is_exit_requested(self):
        """检查是否请求退出

        Returns:
            bool: 是否请求退出
        """
        return self.exit_requested

    def show_cache_stats(self):
        """显示缓存统计信息"""
        from utils.cache_monitor import get_monitor

        monitor = get_monitor()

        # 确保监控线程运行
        if not monitor.monitoring:
            monitor.start_monitoring()

        # 立即持久化当前统计
        for name in monitor.monitored_caches:
            monitor._persist_stats(name)

        # 打印统计
        monitor.print_stats()

        # 提醒用户
        print("\n缓存统计数据已保存到 logs/cache_stats/ 目录")
