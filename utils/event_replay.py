# -*- coding: utf-8 -*-
"""
事件回放控制器 - 用于回放和测试保存的事件序列

本模块提供:
1. 事件序列加载和准备
2. 可控制的事件回放功能
3. 事件回放可视化界面
4. 事件测试和验证功能
"""
import os
import json
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 尝试导入事件序列化模块
try:
    from utils.event_serializer import deserialize_event, format_timestamp
except ImportError:
    # 备用导入路径
    try:
        from event_serializer import deserialize_event, format_timestamp
    except ImportError:
        # 内联实现，确保代码可以独立运行
        def deserialize_event(serialized_event):
            """将序列化的事件字典转换回事件格式"""
            event_type = serialized_event['event_type']
            event_data = serialized_event['data']

            # 确保timestamp字段存在
            if 'timestamp' not in event_data and 'timestamp' in serialized_event:
                event_data['timestamp'] = serialized_event['timestamp']

            return event_type, event_data


        def format_timestamp(timestamp):
            """将时间戳格式化为人类可读的时间字符串"""
            return datetime.fromtimestamp(timestamp).strftime(
                '%Y-%m-%d %H:%M:%S.%f')

# 尝试导入事件系统
try:
    from utils.event_system import get_event_system
except ImportError:
    # 备用导入路径
    try:
        from event_system import get_event_system
    except ImportError:
        # 提供一个简化的获取事件系统的函数
        def get_event_system():
            """获取事件系统实例"""
            # 这里返回None，稍后检查并处理
            return None

# 配置日志记录器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EventReplayController")


class EventReplayController:
    """
    事件回放控制器 - 可加载、控制和回放保存的事件序列

    支持：
    - 加载保存的事件序列
    - 控制回放速度、暂停/继续
    - 单步执行和快进
    - 可视化事件序列
    """

    def __init__(self, event_system=None):
        """
        初始化事件回放控制器

        Args:
            event_system: 可选的事件系统实例，用于发布回放的事件
        """
        # 事件系统
        self.event_system = event_system or get_event_system()

        # 回放控制
        self.events = []  # 加载的事件序列
        self.current_index = 0  # 当前回放位置
        self.is_playing = False  # 是否正在回放
        self.playback_speed = 1.0  # 回放速度倍率
        self.playback_thread = None  # 回放线程

        # 事件过滤
        self.event_filters = {}  # 事件类型过滤器

        # 统计数据
        self.event_stats = {}  # 事件类型统计

        # 回放状态变更回调
        self.state_callbacks = []  # 状态变更回调函数列表

        logger.info("事件回放控制器已初始化")

    def load_events_from_file(self, filepath):
        """
        从文件加载事件序列

        Args:
            filepath: 事件文件路径

        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(filepath):
            logger.error(f"事件文件不存在: {filepath}")
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                serialized_events = json.load(f)

            # 反序列化事件
            self.events = []
            for serialized_event in serialized_events:
                event_type, event_data = deserialize_event(serialized_event)
                self.events.append((event_type, event_data,
                                    serialized_event.get('timestamp', 0)))

            # 按时间戳排序
            self.events.sort(key=lambda x: x[2])

            # 重置回放状态
            self.reset()

            # 计算事件统计
            self._calculate_stats()

            logger.info(f"已从{filepath}加载{len(self.events)}个事件")
            return True

        except Exception as e:
            logger.error(f"加载事件文件失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _calculate_stats(self):
        """计算事件统计数据"""
        self.event_stats = {}

        for event_type, _, _ in self.events:
            if event_type not in self.event_stats:
                self.event_stats[event_type] = 0
            self.event_stats[event_type] += 1

    def reset(self):
        """重置回放状态"""
        self.stop()
        self.current_index = 0

        # 通知状态变更
        self._notify_state_change({
            'action': 'reset',
            'current_index': self.current_index,
            'total_events': len(self.events)
        })

        logger.info("回放状态已重置")

    def play(self):
        """开始或继续回放"""
        if self.is_playing:
            return

        if not self.events:
            logger.warning("没有可回放的事件")
            return

        self.is_playing = True

        # 创建并启动回放线程
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()

        # 通知状态变更
        self._notify_state_change({
            'action': 'play',
            'current_index': self.current_index,
            'total_events': len(self.events)
        })

        logger.info("开始回放事件")

    def pause(self):
        """暂停回放"""
        if not self.is_playing:
            return

        self.is_playing = False

        if self.playback_thread:
            self.playback_thread.join(timeout=0.5)
            self.playback_thread = None

        # 通知状态变更
        self._notify_state_change({
            'action': 'pause',
            'current_index': self.current_index,
            'total_events': len(self.events)
        })

        logger.info("回放已暂停")

    def stop(self):
        """停止回放"""
        # 暂停回放
        self.pause()

        # 重置位置
        self.current_index = 0

        # 通知状态变更
        self._notify_state_change({
            'action': 'stop',
            'current_index': self.current_index,
            'total_events': len(self.events)
        })

        logger.info("回放已停止")

    def step_forward(self):
        """单步前进"""
        if not self.events or self.current_index >= len(self.events):
            return False

        # 暂停当前回放
        was_playing = self.is_playing
        self.pause()

        # 回放单个事件
        self._replay_event(self.current_index)
        self.current_index += 1

        # 通知状态变更
        self._notify_state_change({
            'action': 'step',
            'current_index': self.current_index,
            'total_events': len(self.events)
        })

        # 如果之前在播放，继续播放
        if was_playing:
            self.play()

        return True

    def step_backward(self):
        """单步后退"""
        # 暂停当前回放
        was_playing = self.is_playing
        self.pause()

        if self.current_index > 0:
            self.current_index -= 1

            # 通知状态变更
            self._notify_state_change({
                'action': 'step_back',
                'current_index': self.current_index,
                'total_events': len(self.events)
            })

            # 如果之前在播放，继续播放
            if was_playing:
                self.play()

            return True

        return False

    def fast_forward(self, count=10):
        """快进多个事件"""
        if not self.events:
            return False

        # 暂停当前回放
        was_playing = self.is_playing
        self.pause()

        # 计算目标位置
        target_index = min(self.current_index + count, len(self.events))

        # 回放这些事件
        for i in range(self.current_index, target_index):
            self._replay_event(i)

        self.current_index = target_index

        # 通知状态变更
        self._notify_state_change({
            'action': 'fast_forward',
            'current_index': self.current_index,
            'total_events': len(self.events)
        })

        # 如果之前在播放，继续播放
        if was_playing and self.current_index < len(self.events):
            self.play()

        return True

    def set_playback_speed(self, speed):
        """
        设置回放速度

        Args:
            speed: 回放速度倍率 (0.25, 0.5, 1.0, 2.0, 5.0)
        """
        # 确保速度在有效范围内
        valid_speeds = [0.25, 0.5, 1.0, 2.0, 5.0]
        if speed not in valid_speeds:
            speed = min(valid_speeds, key=lambda x: abs(x - speed))

        self.playback_speed = speed

        # 通知状态变更
        self._notify_state_change({
            'action': 'speed_change',
            'current_index': self.current_index,
            'playback_speed': self.playback_speed
        })

        logger.info(f"回放速度已设置为 {speed}x")

    def add_state_callback(self, callback):
        """
        添加状态变更回调函数

        Args:
            callback: 回调函数，接收状态字典作为参数
        """
        if callable(callback) and callback not in self.state_callbacks:
            self.state_callbacks.append(callback)

    def remove_state_callback(self, callback):
        """移除状态变更回调函数"""
        if callback in self.state_callbacks:
            self.state_callbacks.remove(callback)

    def _notify_state_change(self, state_data):
        """通知所有回调函数状态已变更"""
        for callback in self.state_callbacks:
            try:
                callback(state_data)
            except Exception as e:
                logger.error(f"执行状态回调时出错: {e}")

    def set_event_filter(self, event_type, enabled=True):
        """
        设置事件类型过滤器

        Args:
            event_type: 事件类型
            enabled: 是否启用此类型事件
        """
        self.event_filters[event_type] = enabled

        # 通知状态变更
        self._notify_state_change({
            'action': 'filter_change',
            'event_type': event_type,
            'enabled': enabled
        })

        logger.info(f"事件过滤器已更新: {event_type} -> {enabled}")

    def clear_event_filters(self):
        """清除所有事件过滤器"""
        self.event_filters = {}

        # 通知状态变更
        self._notify_state_change({
            'action': 'filters_cleared'
        })

        logger.info("所有事件过滤器已清除")

    def _playback_loop(self):
        """回放循环 - 在单独线程中运行"""
        try:
            while self.is_playing and self.current_index < len(self.events):
                # 回放当前事件
                self._replay_event(self.current_index)

                # 前进到下一个事件
                self.current_index += 1

                # 通知状态变更
                self._notify_state_change({
                    'action': 'progress',
                    'current_index': self.current_index,
                    'total_events': len(self.events)
                })

                # 检查是否已到达末尾
                if self.current_index >= len(self.events):
                    self.pause()
                    logger.info("回放已完成")

                    # 通知回放完成
                    self._notify_state_change({
                        'action': 'completed',
                        'current_index': self.current_index,
                        'total_events': len(self.events)
                    })
                    break

                # 计算下一个事件的延迟
                if self.current_index < len(self.events):
                    current_time = self.events[self.current_index - 1][2]
                    next_time = self.events[self.current_index][2]
                    delay = (next_time - current_time) / self.playback_speed
                    delay = max(0, min(delay, 2.0))  # 限制最大延迟为2秒

                    # 等待适当的时间
                    for _ in range(int(delay * 10)):
                        if not self.is_playing:
                            break
                        time.sleep(0.1)

        except Exception as e:
            logger.error(f"回放循环出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.is_playing = False

    def _replay_event(self, index):
        """
        回放指定索引的事件

        Args:
            index: 事件索引
        """
        if index < 0 or index >= len(self.events):
            return False

        event_type, event_data, timestamp = self.events[index]

        # 检查事件过滤器
        if event_type in self.event_filters and not self.event_filters[
            event_type]:
            logger.debug(f"事件已过滤: {event_type}")
            return False

        # 如果有事件系统，发布事件
        if self.event_system:
            try:
                logger.debug(f"回放事件: {event_type}")
                self.event_system.publish(event_type, event_data)
            except Exception as e:
                logger.error(f"发布事件时出错: {e}")

        # 打印事件信息
        formatted_time = format_timestamp(timestamp)
        logger.info(f"[{formatted_time}] {event_type}: {event_data}")

        return True

    def get_current_event(self):
        """
        获取当前位置的事件

        Returns:
            tuple or None: (event_type, event_data, timestamp) 或 None
        """
        if not self.events or self.current_index >= len(self.events):
            return None

        return self.events[self.current_index]

    def jump_to_index(self, index):
        """
        跳转到指定索引

        Args:
            index: 目标索引

        Returns:
            bool: 是否成功跳转
        """
        if not self.events:
            return False

        # 确保索引在有效范围内
        index = max(0, min(index, len(self.events)))

        # 暂停当前回放
        was_playing = self.is_playing
        self.pause()

        # 设置新位置
        self.current_index = index

        # 通知状态变更
        self._notify_state_change({
            'action': 'jump',
            'current_index': self.current_index,
            'total_events': len(self.events)
        })

        # 如果之前在播放，继续播放
        if was_playing:
            self.play()

        return True

    def jump_to_event_type(self, event_type, direction='forward'):
        """
        跳转到指定类型的事件

        Args:
            event_type: 目标事件类型
            direction: 搜索方向 ('forward' 或 'backward')

        Returns:
            bool: 是否成功跳转
        """
        if not self.events:
            return False

        # 暂停当前回放
        was_playing = self.is_playing
        self.pause()

        # 搜索指定类型的事件
        found_index = -1

        if direction == 'forward':
            # 向前搜索
            for i in range(self.current_index, len(self.events)):
                if self.events[i][0] == event_type:
                    found_index = i
                    break
        else:
            # 向后搜索
            for i in range(self.current_index - 1, -1, -1):
                if self.events[i][0] == event_type:
                    found_index = i
                    break

        # 如果找到匹配的事件，跳转到该位置
        if found_index >= 0:
            self.current_index = found_index

            # 通知状态变更
            self._notify_state_change({
                'action': 'jump_to_type',
                'current_index': self.current_index,
                'event_type': event_type,
                'total_events': len(self.events)
            })

            # 如果之前在播放，继续播放
            if was_playing:
                self.play()

            return True

        return False

    def get_event_types(self):
        """
        获取所有事件类型及其数量

        Returns:
            dict: 事件类型 -> 数量的映射
        """
        return dict(self.event_stats)


class EventReplayGUI:
    """
    事件回放图形界面 - 提供用户友好的事件回放控制和可视化
    """

    def __init__(self, controller=None):
        """
        初始化事件回放GUI

        Args:
            controller: 可选的事件回放控制器，如果为None则创建新实例
        """
        # 创建控制器
        self.controller = controller or EventReplayController()

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("事件回放控制器")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # 设置事件处理回调
        self.controller.add_state_callback(self._on_state_change)

        # 创建UI组件
        self._create_ui()

        # 更新状态
        self._update_status()

        logger.info("事件回放GUI已初始化")

    def _create_ui(self):
        """创建用户界面"""
        # 主布局
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 顶部控制栏
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="5")
        control_frame.pack(fill=tk.X, pady=5)

        # 控制按钮
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(buttons_frame, text="加载事件",
                                      command=self._on_load)
        self.load_button.pack(side=tk.LEFT, padx=2)

        self.play_button = ttk.Button(buttons_frame, text="播放",
                                      command=self._on_play)
        self.play_button.pack(side=tk.LEFT, padx=2)

        self.pause_button = ttk.Button(buttons_frame, text="暂停",
                                       command=self._on_pause)
        self.pause_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = ttk.Button(buttons_frame, text="停止",
                                      command=self._on_stop)
        self.stop_button.pack(side=tk.LEFT, padx=2)

        self.step_back_button = ttk.Button(buttons_frame, text="<<",
                                           command=self._on_step_back)
        self.step_back_button.pack(side=tk.LEFT, padx=2)

        self.step_button = ttk.Button(buttons_frame, text="单步",
                                      command=self._on_step)
        self.step_button.pack(side=tk.LEFT, padx=2)

        self.ff_button = ttk.Button(buttons_frame, text=">>",
                                    command=self._on_fast_forward)
        self.ff_button.pack(side=tk.LEFT, padx=2)

        # 速度控制
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(speed_frame, text="速度:").pack(side=tk.LEFT)

        self.speed_var = tk.StringVar(value="1.0x")
        speed_options = ["0.25x", "0.5x", "1.0x", "2.0x", "5.0x"]
        self.speed_menu = ttk.OptionMenu(
            speed_frame,
            self.speed_var,
            "1.0x",
            *speed_options,
            command=self._on_speed_change
        )
        self.speed_menu.pack(side=tk.LEFT, padx=5)

        # 状态显示
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(side=tk.RIGHT, padx=10)

        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.RIGHT)

        # 内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 左侧事件列表面板
        events_frame = ttk.LabelFrame(content_frame, text="事件列表",
                                      padding="5")
        events_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 事件列表控件
        columns = ("索引", "时间", "事件类型", "数据预览")
        self.event_tree = ttk.Treeview(events_frame, columns=columns,
                                       show="headings")

        # 设置列标题
        self.event_tree.heading("索引", text="索引")
        self.event_tree.heading("时间", text="时间")
        self.event_tree.heading("事件类型", text="事件类型")
        self.event_tree.heading("数据预览", text="数据预览")

        # 设置列宽
        self.event_tree.column("索引", width=50, anchor="center")
        self.event_tree.column("时间", width=150)
        self.event_tree.column("事件类型", width=150)
        self.event_tree.column("数据预览", width=300)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(events_frame, orient="vertical",
                                  command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=scrollbar.set)

        # 放置列表和滚动条
        self.event_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定选择事件
        self.event_tree.bind("<<TreeviewSelect>>", self._on_event_select)

        # 右侧事件详情面板
        details_frame = ttk.LabelFrame(content_frame, text="事件详情",
                                       padding="5")
        details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                           padx=(5, 0))

        # 事件详情文本控件
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, width=40)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical",
                                          command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 底部过滤和统计面板
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, pady=5)

        # 过滤器面板
        filter_frame = ttk.LabelFrame(bottom_frame, text="事件过滤",
                                      padding="5")
        filter_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 过滤器控件容器（使用Canvas使其可滚动）
        filter_canvas = tk.Canvas(filter_frame)
        filter_scrollbar = ttk.Scrollbar(filter_frame, orient="vertical",
                                         command=filter_canvas.yview)
        filter_canvas.configure(yscrollcommand=filter_scrollbar.set)

        filter_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.filters_frame = ttk.Frame(filter_canvas)
        filter_canvas.create_window((0, 0), window=self.filters_frame,
                                    anchor="nw")

        self.filters_frame.bind("<Configure>",
                                lambda e: filter_canvas.configure(
                                    scrollregion=filter_canvas.bbox("all")))

        # 统计面板
        stats_frame = ttk.LabelFrame(bottom_frame, text="事件统计", padding="5")
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 创建统计图表
        self.figure = plt.Figure(figsize=(4, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, stats_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 底部进度条
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X)

        # 更新界面状态
        self._update_ui_state()

    def _on_load(self):
        """加载事件文件按钮回调"""
        filepath = filedialog.askopenfilename(
            title="选择事件文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )

        if not filepath:
            return

        # 加载事件文件
        if self.controller.load_events_from_file(filepath):
            self._populate_event_list()
            self._update_filters()
            self._update_stats()
            messagebox.showinfo("加载完成",
                                f"已加载 {len(self.controller.events)} 个事件")
        else:
            messagebox.showerror("加载失败",
                                 "无法加载事件文件，请查看日志获取详情")

    def _on_play(self):
        """播放按钮回调"""
        self.controller.play()

    def _on_pause(self):
        """暂停按钮回调"""
        self.controller.pause()

    def _on_stop(self):
        """停止按钮回调"""
        self.controller.stop()

    def _on_step(self):
        """单步按钮回调"""
        self.controller.step_forward()

    def _on_step_back(self):
        """后退按钮回调"""
        self.controller.step_backward()

    def _on_fast_forward(self):
        """快进按钮回调"""
        self.controller.fast_forward(10)

    def _on_speed_change(self, *args):
        """速度变更回调"""
        speed_str = self.speed_var.get()
        speed = float(speed_str.rstrip('x'))
        self.controller.set_playback_speed(speed)

    def _on_event_select(self, event):
        """事件选择回调"""
        selected = self.event_tree.selection()
        if not selected:
            return

        # 获取选中项的索引
        item = self.event_tree.item(selected[0])
        index = int(item['values'][0])

        # 跳转到选中的事件
        self.controller.jump_to_index(index)

        # 显示事件详情
        self._show_event_details(index)

    def _on_filter_change(self, event_type, var):
        """过滤器变更回调"""
        enabled = var.get()
        self.controller.set_event_filter(event_type, enabled)

    def _on_state_change(self, state):
        """状态变更回调"""
        action = state.get('action')

        # 更新进度
        if 'current_index' in state and 'total_events' in state:
            current = state['current_index']
            total = state['total_events']
            if total > 0:
                progress = current / total * 100
                self.progress_var.set(progress)

                # 如果位置变化，选中对应的事件
                self._select_current_event()

        # 更新UI状态
        self._update_ui_state()

        # 更新状态标签
        if action == 'play':
            self.status_label.config(text="正在播放")
        elif action == 'pause':
            self.status_label.config(text="已暂停")
        elif action == 'stop':
            self.status_label.config(text="已停止")
        elif action == 'completed':
            self.status_label.config(text="播放完成")

    def _update_ui_state(self):
        """更新UI控件状态"""
        # 获取控制器状态
        is_playing = self.controller.is_playing
        has_events = len(self.controller.events) > 0
        current_index = self.controller.current_index
        total_events = len(self.controller.events)

        # 更新按钮状态
        self.play_button.config(
            state=tk.NORMAL if has_events and not is_playing else tk.DISABLED)
        self.pause_button.config(
            state=tk.NORMAL if is_playing else tk.DISABLED)
        self.stop_button.config(
            state=tk.NORMAL if has_events else tk.DISABLED)
        self.step_button.config(
            state=tk.NORMAL if has_events and current_index < total_events else tk.DISABLED)
        self.step_back_button.config(
            state=tk.NORMAL if has_events and current_index > 0 else tk.DISABLED)
        self.ff_button.config(
            state=tk.NORMAL if has_events and current_index < total_events else tk.DISABLED)

    def _update_status(self):
        """更新状态信息"""
        has_events = len(self.controller.events) > 0

        if not has_events:
            self.status_label.config(text="未加载事件")
        elif self.controller.is_playing:
            self.status_label.config(text="正在播放")
        else:
            current = self.controller.current_index
            total = len(self.controller.events)
            self.status_label.config(text=f"就绪 ({current}/{total})")

    def _populate_event_list(self):
        """填充事件列表"""
        # 清空现有项
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)

        # 添加新项
        for i, (event_type, event_data, timestamp) in enumerate(
                self.controller.events):
            # 格式化时间
            time_str = format_timestamp(timestamp)

            # 准备数据预览
            data_preview = str(event_data)
            if len(data_preview) > 50:
                data_preview = data_preview[:47] + "..."

            # 添加到树形控件
            self.event_tree.insert("", "end", values=(
            i, time_str, event_type, data_preview))

    def _select_current_event(self):
        """选中当前事件"""
        if not self.controller.events:
            return

        current = self.controller.current_index
        if current >= len(self.controller.events):
            return

        # 获取所有项
        items = self.event_tree.get_children()
        if current < len(items):
            # 选中当前项
            item = items[current]
            self.event_tree.selection_set(item)
            self.event_tree.see(item)

            # 显示事件详情
            self._show_event_details(current)

    def _show_event_details(self, index):
        """显示事件详情"""
        if index < 0 or index >= len(self.controller.events):
            return

        event_type, event_data, timestamp = self.controller.events[index]

        # 清空文本区域
        self.details_text.delete("1.0", tk.END)

        # 格式化时间
        time_str = format_timestamp(timestamp)

        # 添加事件详情
        self.details_text.insert(tk.END, f"事件类型: {event_type}\n\n")
        self.details_text.insert(tk.END, f"时间戳: {time_str}\n\n")
        self.details_text.insert(tk.END, f"数据:\n\n")

        # 格式化JSON数据
        try:
            formatted_data = json.dumps(event_data, indent=2,
                                        ensure_ascii=False)
            self.details_text.insert(tk.END, formatted_data)
        except:
            self.details_text.insert(tk.END, str(event_data))

    def _update_filters(self):
        """更新过滤器面板"""
        # 清空现有过滤器
        for widget in self.filters_frame.winfo_children():
            widget.destroy()

        # 获取所有事件类型
        event_types = self.controller.get_event_types()

        # 创建过滤器控件
        self.filter_vars = {}

        row = 0
        col = 0
        max_cols = 2  # 每行最多显示的过滤器数量

        for event_type, count in sorted(event_types.items()):
            # 创建变量
            var = tk.BooleanVar(value=True)
            self.filter_vars[event_type] = var

            # 创建复选框
            frame = ttk.Frame(self.filters_frame)
            frame.grid(row=row, column=col, sticky="w", padx=5, pady=2)

            checkbox = ttk.Checkbutton(
                frame,
                text=f"{event_type} ({count})",
                variable=var,
                command=lambda et=event_type, v=var: self._on_filter_change(
                    et, v)
            )
            checkbox.pack(side=tk.LEFT)

            # 更新行列位置
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _update_stats(self):
        """更新统计图表"""
        # 获取事件统计
        event_stats = self.controller.get_event_types()

        if not event_stats:
            return

        # 清除之前的图表
        self.ax.clear()

        # 准备数据
        types = []
        counts = []

        # 只显示前10种事件类型，避免图表过于拥挤
        for event_type, count in sorted(event_stats.items(),
                                        key=lambda x: x[1], reverse=True)[
                                 :10]:
            # 截断过长的事件类型名称
            if len(event_type) > 15:
                event_type = event_type[:12] + "..."
            types.append(event_type)
            counts.append(count)

        # 绘制水平条形图
        y_pos = range(len(types))
        self.ax.barh(y_pos, counts)
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(types)
        self.ax.invert_yaxis()  # 将最高频的类型显示在顶部
        self.ax.set_xlabel('事件数量')
        self.ax.set_title('事件类型分布')

        # 在每个条形上显示具体数值
        for i, v in enumerate(counts):
            self.ax.text(v + 0.1, i, str(v), va='center')

        # 更新图表
        self.figure.tight_layout()
        self.canvas.draw()

    def run(self):
        """运行GUI主循环"""
        self.root.mainloop()

class EventReplayTool:
    """
    事件回放工具 - 集成命令行和GUI接口

    可以从命令行运行，也可以作为模块导入
    """

    @classmethod
    def run_gui(cls, event_file=None):
        """
        运行GUI模式

        Args:
            event_file: 可选的事件文件路径
        """
        controller = EventReplayController()
        gui = EventReplayGUI(controller)

        # 如果提供了事件文件，加载它
        if event_file:
            controller.load_events_from_file(event_file)

        # 运行GUI
        gui.run()

    @classmethod
    def run_cli(cls, event_file, output=None):
        """
        运行命令行模式

        Args:
            event_file: 事件文件路径
            output: 可选的输出文件路径
        """
        controller = EventReplayController()

        # 加载事件文件
        if not controller.load_events_from_file(event_file):
            print(f"错误: 无法加载事件文件 {event_file}")
            return

        # 打印事件统计
        event_stats = controller.get_event_types()
        print("\n事件类型统计:")
        for event_type, count in sorted(event_stats.items(),
                                        key=lambda x: x[1], reverse=True):
            print(f"  {event_type}: {count}")

        total_events = len(controller.events)
        print(f"\n总共 {total_events} 个事件")

        # 如果指定了输出文件，将事件列表导出
        if output:
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write("索引,时间,事件类型,数据\n")
                    for i, (event_type, event_data, timestamp) in enumerate(
                            controller.events):
                        time_str = format_timestamp(timestamp)
                        data_str = json.dumps(event_data,
                                              ensure_ascii=False).replace(
                            '"', '""')
                        f.write(
                            f"{i},\"{time_str}\",\"{event_type}\",\"{data_str}\"\n")
                print(f"\n事件列表已导出到 {output}")
            except Exception as e:
                print(f"导出事件列表时出错: {e}")

        # 交互式命令处理
        print("\n输入命令 (help显示帮助，exit退出):")
        while True:
            try:
                cmd = input("> ").strip().lower()

                if cmd in ['exit', 'quit', 'q']:
                    break
                elif cmd in ['help', 'h', '?']:
                    print("可用命令:")
                    print("  help, h, ? - 显示帮助")
                    print("  exit, quit, q - 退出程序")
                    print("  list [n] - 列出前n个事件 (默认10)")
                    print("  find <类型> - 查找指定类型的事件")
                    print("  info - 显示事件文件信息")
                    print("  gui - 启动图形界面")
                elif cmd.startswith('list'):
                    parts = cmd.split()
                    n = 10  # 默认显示10个事件
                    if len(parts) > 1:
                        try:
                            n = int(parts[1])
                        except:
                            pass

                    print(f"\n事件列表 (前{min(n, total_events)}个):")
                    for i, (event_type, event_data, timestamp) in enumerate(
                            controller.events[:n]):
                        time_str = format_timestamp(timestamp)
                        print(f"{i}: [{time_str}] {event_type}")
                elif cmd.startswith('find'):
                    parts = cmd.split()
                    if len(parts) > 1:
                        search_type = parts[1]
                        found = False

                        print(f"\n查找事件类型: {search_type}")
                        for i, (
                        event_type, event_data, timestamp) in enumerate(
                                controller.events):
                            if search_type.lower() in event_type.lower():
                                time_str = format_timestamp(timestamp)
                                print(f"{i}: [{time_str}] {event_type}")
                                found = True

                        if not found:
                            print(f"未找到类型为 '{search_type}' 的事件")
                    else:
                        print("请指定要查找的事件类型")
                elif cmd == 'info':
                    print("\n事件文件信息:")
                    print(f"  文件: {event_file}")
                    print(f"  事件总数: {total_events}")
                    print(f"  事件类型数: {len(event_stats)}")

                    if controller.events:
                        first_time = format_timestamp(
                            controller.events[0][2])
                        last_time = format_timestamp(
                            controller.events[-1][2])
                        print(f"  时间范围: {first_time} 到 {last_time}")
                elif cmd == 'gui':
                    # 启动GUI
                    print("启动图形界面...")
                    cls.run_gui(event_file)
                    break
                else:
                    print(f"未知命令: {cmd}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"命令执行错误: {e}")

def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="事件回放工具")
    parser.add_argument("event_file", nargs="?", help="事件文件路径")
    parser.add_argument("--gui", action="store_true", help="启动图形界面")
    parser.add_argument("--output", "-o", help="输出文件路径")

    args = parser.parse_args()

    if args.gui or not args.event_file:
        # GUI模式
        EventReplayTool.run_gui(args.event_file)
    else:
        # 命令行模式
        EventReplayTool.run_cli(args.event_file, args.output)

if __name__ == "__main__":
    main()
