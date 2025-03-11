# -*- coding: utf-8 -*-
"""
增强型显示管理器 - 支持事件驱动的UI更新

此模块扩展了DisplayManager类，添加了与UI事件系统的深度集成，
使UI更新可以由事件驱动，而不是传统的轮询方式。
"""
import cv2
import logging
import numpy as np
import time
from utils.logger_config import get_logger
from utils.ui_events import (
    UIEventTypes,
    UIEventPublisher,
    UIEventSubscriber,
    UIEventDrivenComponent
)

logger = get_logger("EnhancedDisplayManager")


class EnhancedDisplayManager(UIEventDrivenComponent):
    """
    增强型显示管理器 - 支持事件驱动的UI更新

    基于原始DisplayManager，但添加了事件驱动的UI更新机制，
    可以响应各种事件自动更新UI，减少轮询和手动更新的需要。
    """

    def __init__(self, visualizer, event_system=None):
        """
        初始化增强型显示管理器

        Args:
            visualizer: 可视化器
            event_system: 事件系统
        """
        # 调用基类初始化
        super().__init__("display_manager")

        # 基本属性
        self.visualizer = visualizer
        self.last_room_viz = None
        self.last_camera_viz = None
        self.debug_mode = True
        self.show_advanced_info = False
        self.windows_created = False
        self.last_update_time = time.time()
        self.update_interval = 1.0 / 30  # 最大30FPS更新频率

        # 上次渲染时间和FPS
        self.last_fps_update_time = time.time()
        self.frame_count = 0
        self.current_fps = 0

        # 当前要显示的数据
        self.current_frame = None
        self.current_room_position = None
        self.current_depth = None
        self.current_action = None
        self.current_person = None
        self.current_system_state = "初始化中"
        self.current_debug_info = {}

        # 显示选项
        self.display_options = {
            "show_fps": True,
            "show_skeleton": True,
            "show_bbox": True,
            "show_trail": True,
            "show_action": True
        }

        # 功能开关状态
        self.feature_states = {
            "mediapipe": False,
            "ml_model": False,
            "dtw": False,
            "threading": False,
            "async": False
        }

        # 键盘快捷键说明
        self.key_hints = {
            'q': '退出',
            'd': '调试信息',
            'r': '重新校准',
            'f': 'FPS显示',
            's': '骨架显示',
            'b': '边界框显示',
            't': '轨迹显示',
            'a': '动作显示',
            'h': '帮助',
            'm': 'MediaPipe',
            'l': '机器学习模型',
            'w': 'DTW算法',
            'p': '多线程处理',
            'y': '异步管道'
        }

        # 通知系统
        self.notifications = []  # [(消息, 级别, 显示时间, 结束时间)]
        self.notification_duration = 3.0  # 默认通知显示3秒

        logger.info("增强型显示管理器已初始化")

    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 订阅各种事件类型

        # 人体检测事件
        self.subscriber.subscribe(
            "person_detected",
            self.on_person_detected
        )

        # 动作识别事件
        self.subscriber.subscribe(
            "action_recognized",
            self.on_action_recognized
        )

        # 位置映射事件
        self.subscriber.subscribe(
            "position_mapped",
            self.on_position_mapped
        )

        # 特性开关事件
        self.subscriber.subscribe(
            UIEventTypes.FEATURE_TOGGLED,
            self.on_feature_toggled
        )

        # 状态变化事件
        self.subscriber.subscribe(
            "system_state_changed",
            self.on_system_state_changed
        )

        # FPS更新事件
        self.subscriber.subscribe(
            UIEventTypes.FPS_UPDATED,
            self.on_fps_updated
        )

        # 通知事件
        self.subscriber.subscribe_to_notifications(
            self.handle_notification
        )

        # 性能警告事件
        self.subscriber.subscribe(
            UIEventTypes.PERFORMANCE_WARNING,
            self.handle_performance_warning
        )

        # 显示选项变更事件
        self.subscriber.subscribe(
            UIEventTypes.OPTION_TOGGLED,
            self.handle_ui_state_change
        )

        logger.info("显示管理器已设置事件处理器")

    def on_person_detected(self, data):
        """人体检测事件处理"""
        if 'person' in data:
            self.current_person = data['person']
            # 触发显示更新
            self.update()

    def on_action_recognized(self, data):
        """动作识别事件处理"""
        if 'action' in data:
            self.current_action = data['action']
            # 触发显示更新
            self.update()

    def on_position_mapped(self, data):
        """位置映射事件处理"""
        if 'position' in data:
            self.current_room_position = data['position']
            if 'depth' in data:
                self.current_depth = data['depth']

            # 更新轨迹点
            if self.visualizer and hasattr(self.visualizer, 'add_trail_point'):
                self.visualizer.add_trail_point(*self.current_room_position)

            # 触发显示更新
            self.update()

    def on_feature_toggled(self, data):
        """功能开关事件处理"""
        if 'feature_name' in data and 'state' in data:
            self.update_feature_state(data['feature_name'], data['state'])
            # 触发显示更新
            self.update()

    def on_system_state_changed(self, data):
        """系统状态变化事件处理"""
        if 'state' in data:
            self.current_system_state = data['state']
            # 触发显示更新
            self.update()

    def on_fps_updated(self, data):
        """FPS更新事件处理"""
        if 'fps' in data:
            self.current_fps = data['fps']
            # 不需要触发更新，因为FPS信息会在下一次正常更新时显示

    def handle_notification(self, data):
        """处理通知事件"""
        message = data.get('message', '')
        level = data.get('level', 'info')
        duration = data.get('duration', self.notification_duration)

        # 添加到通知队列
        self.notifications.append(
            (message, level, duration, time.time() + duration)
        )

        # 触发显示更新
        self.update()

    def handle_performance_warning(self, data):
        """处理性能警告事件"""
        warning_type = data.get('warning_type', 'unknown')

        if warning_type == 'low_fps' and 'fps' in data:
            fps = data['fps']
            # 添加性能警告通知
            self.notifications.append(
                (f"性能警告: FPS={fps:.1f}", "warning", 2.0, time.time() + 2.0)
            )

            # 触发显示更新
            self.update()

    def create_windows(self):
        """创建并配置显示窗口"""
        if not self.windows_created:
            try:
                cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Camera View', 800, 600)
                cv2.namedWindow('Room Position', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Room Position', 800, 600)
                self.windows_created = True
                logger.info("创建显示窗口")

                # 发布窗口创建事件
                self.publisher.publish_window_event(
                    UIEventTypes.WINDOW_CREATED,
                    "Camera View",
                    (800, 600)
                )

                self.publisher.publish_window_event(
                    UIEventTypes.WINDOW_CREATED,
                    "Room Position",
                    (800, 600)
                )

                return True
            except Exception as e:
                logger.error(f"创建窗口失败: {e}")
                return False

        return True

    def update_frame(self, frame):
        """
        更新当前帧

        Args:
            frame: 新的相机帧
        """
        if frame is not None and isinstance(frame, np.ndarray):
            self.current_frame = frame.copy()
            self.frame_count += 1

            # 检查是否需要更新FPS
            current_time = time.time()
            elapsed = current_time - self.last_fps_update_time

            if elapsed >= 1.0:  # 每秒更新一次FPS
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_update_time = current_time

                # 发布FPS更新事件
                self.publisher.publish_fps_update(
                    self.current_fps,
                    1000.0 / max(1, self.current_fps)  # 毫秒/帧
                )

            # 更新UI
            self.update()

    def update_debug_info(self, debug_info):
        """
        更新调试信息

        Args:
            debug_info: 调试信息字典
        """
        if debug_info and isinstance(debug_info, dict):
            self.current_debug_info = debug_info

            # 发布调试信息更新事件
            self.publisher.publish_display_event(
                UIEventTypes.DEBUG_INFO_UPDATED,
                debug_info=self.current_debug_info
            )

            # 更新UI
            self.update()
