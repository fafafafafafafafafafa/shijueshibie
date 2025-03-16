# -*- coding: utf-8 -*-
"""
显示管理器模块 - 负责处理显示和用户界面

增强版: 支持事件驱动的UI更新，同时保持原有接口不变。
"""
import cv2
import logging
import numpy as np
import time
from utils.logger_config import get_logger
from utils.ui_events import (
    UIEventTypes,
    UIEventPublisher,
    UIEventSubscriber
)
from utils.conditional_events import where
from core.component_interface import BaseLifecycleComponent, LifecycleComponentInterface
from core.event_bus import get_event_bus
from core.component_lifecycle import LifecycleState


logger = get_logger("DisplayManager")


class DisplayManager(BaseLifecycleComponent):
    """
    显示管理器 - 负责处理显示和用户界面

    支持多种显示模式和调试信息，内部采用事件驱动机制。
    """

    def __init__(self, visualizer=None, component_id="display_manager", component_type="UI"):
        """
        初始化显示管理器

        Args:
            visualizer: 可视化器实例
        """
        # 初始化基础生命周期组件
        super().__init__(component_id, component_type)

        # 基本属性
        self.visualizer =visualizer
        self.last_room_viz = None

        self.debug_mode = True
        self.show_advanced_info = False
        self.windows_created = False
        self.logger = logger

        # 事件系统支持
        self.event_system = None
        self.publisher = UIEventPublisher.get_instance("display_manager")
        self.subscriber = UIEventSubscriber.get_instance("display_manager")

        # 性能监控
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_update_time = time.time()
        self.update_interval = 1.0 / 30  # 最大30FPS更新频率

        # 当前状态
        self.current_frame = None
        self.current_room_viz = None
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

        # 添加功能开关状态显示
        self.feature_states = {
            "mediapipe": False,
            "ml_model": False,
            "dtw": False,
            "threading": False,
            "async": False
        }

        # 键盘快捷键说明
        self.key_hints = {
            'q': '退出程序',
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

        # 获取事件总线
        self.event_bus = get_event_bus()

        logger.info("显示管理器已初始化")

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

    def force_refresh(self):
        """强制刷新UI显示"""
        self.logger.info("强制刷新UI")

        # 重置最后更新时间
        self.last_update_time = 0

        # 立即触发UI更新
        try:
            # 如果当前有有效帧，重新显示
            if self.current_frame is not None:
                camera_viz = self._create_camera_visualization()
                if camera_viz is not None:
                    cv2.imshow('Camera View', camera_viz)

            # 如果有房间视图，重新显示
            if self.current_room_viz is not None:
                cv2.imshow('Room Position', self.current_room_viz)

            # 处理按键
            cv2.waitKey(1)
        except Exception as e:
            self.logger.error(f"强制刷新UI时出错: {e}")

    def set_visualizer(self, visualizer):
        """设置可视化器"""
        self.visualizer = visualizer

    # 实现生命周期方法
    def _do_initialize(self) -> bool:
        """执行初始化逻辑"""
        try:
            # 创建窗口
            self.create_windows()
            # 设置事件处理器
            self._setup_event_handlers()

            logger.info("显示管理器已初始化")
            return True
        except Exception as e:
            logger.error(f"初始化显示管理器时出错: {e}")
            return False

    def _do_start(self) -> bool:
        """执行启动逻辑"""
        try:
            # 发布组件启动事件
            if self.publisher:
                # 使用已存在的方法，而不是尝试使用不存在的属性
                self.publisher.publish_notification(
                    "显示管理器已启动",
                    level="info"
                )

            logger.info("显示管理器已启动")
            return True
        except Exception as e:
            logger.error(f"启动显示管理器时出错: {e}")
            return False

    def _do_stop(self) -> bool:
        """执行停止逻辑"""
        try:
            # 取消所有事件订阅
            if hasattr(self.subscriber, 'unsubscribe_all'):
                self.subscriber.unsubscribe_all()

            logger.info("显示管理器已停止")
            return True
        except Exception as e:
            logger.error(f"停止显示管理器时出错: {e}")
            return False

    def _do_destroy(self) -> bool:
        """执行销毁逻辑"""
        return self.cleanup()



    def create_windows(self):
        """创建并配置显示窗口"""
        if not self.windows_created:
            try:
                cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Camera View', 800, 600)
                cv2.namedWindow('Room Position', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Room Position', 800, 600)
                self.windows_created = True
                self.logger.info("创建显示窗口")

                # 发布窗口创建事件
                if self.publisher:
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
                self.logger.error(f"创建窗口失败: {e}")
                return False

        return True

    def display_frames(self, camera_view, room_view):
        """
        显示相机视图和房间视图

        Args:
            camera_view: 相机视图帧
            room_view: 房间平面图视图
        """
        try:
            # 更新当前帧
            if camera_view is not None and isinstance(camera_view, np.ndarray):
                self.current_frame = camera_view.copy()

                # 更新FPS计数
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_time

                # 每秒更新一次FPS
                if elapsed >= 1.0:
                    self.current_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_time = current_time

                    # 通过事件系统发布FPS更新
                    if self.publisher:
                        self.publisher.publish_fps_update(self.current_fps)

            # 检查窗口是否已创建
            if not self.windows_created:
                self.create_windows()

            # 检查相机视图是否有效
            if not isinstance(camera_view, np.ndarray):
                self.logger.warning(
                    f"相机视图类型错误: {type(camera_view)}，跳过显示")
                return

            if camera_view.size == 0 or camera_view.shape[0] <= 0 or \
                    camera_view.shape[1] <= 0:
                self.logger.warning("相机视图尺寸无效，跳过显示")
                return

            # 更新房间视图
            if room_view is not None and isinstance(room_view, np.ndarray):
                self.current_room_viz = room_view.copy()
            else:
                # 如果房间视图无效，尝试使用上次的有效视图
                self.current_room_viz = self.get_last_room_viz()

            # 检查房间视图是否有效
            if self.current_room_viz is None or self.current_room_viz.size == 0 or \
                    self.current_room_viz.shape[0] <= 0 or \
                    self.current_room_viz.shape[1] <= 0:
                self.logger.warning("房间视图无效，跳过显示")
                return

            # 创建UI显示
            camera_viz = self._create_camera_visualization()

            # 显示图像
            cv2.imshow('Camera View', camera_viz)
            cv2.imshow('Room Position', self.current_room_viz)

            # 保存有效房间视图
            self.last_room_viz = self.current_room_viz.copy()

            # 更新时间戳
            self.last_update_time = time.time()

            # 发布渲染完成事件
            if self.publisher:
                self.publisher.publish_display_event(
                    UIEventTypes.RENDER_COMPLETED,
                    camera_frame=True,
                    room_frame=True
                )

        except Exception as e:
            self.logger.error(f"显示帧错误: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def draw_debug_info(self, frame, fps, system_state, debug_info=None):
        """
        绘制调试信息

        Args:
            frame: 输入帧
            fps: 当前帧率
            system_state: 系统状态
            debug_info: 附加调试信息

        Returns:
            ndarray: 带调试信息的帧
        """
        if not self.debug_mode or not isinstance(frame, np.ndarray):
            return frame

        # 更新内部状态
        self.current_fps = fps
        self.current_system_state = system_state
        self.current_debug_info = debug_info or {}

        # 发布调试信息更新事件
        if self.publisher:
            self.publisher.publish_display_event(
                UIEventTypes.DEBUG_INFO_UPDATED,
                debug_info=self.current_debug_info
            )

        # 创建帧副本
        frame_viz = frame.copy()

        # 基本信息: FPS和状态
        if self.display_options["show_fps"]:
            cv2.putText(frame_viz, f"FPS: {fps:.1f}",
                        (frame_viz.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame_viz, f"Status: {system_state}",
                    (frame_viz.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 添加功能开关状态
        feature_y = 90
        for name, enabled in self.feature_states.items():
            color = (0, 255, 0) if enabled else (0, 0, 255)  # 绿色为开启，红色为关闭
            cv2.putText(frame_viz, f"{name}: {'ON' if enabled else 'OFF'}",
                        (frame_viz.shape[1] - 150, feature_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            feature_y += 25

        # 添加调试信息
        if self.show_advanced_info and debug_info:
            y_pos = feature_y + 10  # 在功能状态之后显示其他调试信息
            for key, value in debug_info.items():
                text = f"{key}: {value}"
                if isinstance(value, tuple) and len(value) == 2:
                    text = f"{key}: ({value[0]:.1f}, {value[1]:.1f})"
                elif isinstance(value, (int, float)):
                    text = f"{key}: {value:.2f}"

                cv2.putText(frame_viz, text,
                            (frame_viz.shape[1] - 250, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos += 25

        # 绘制通知
        self._draw_notifications(frame_viz)

        # 绘制快捷键提示
        help_bg_height = 30
        help_bg = frame_viz.copy()
        cv2.rectangle(help_bg,
                      (0, frame_viz.shape[0] - help_bg_height),
                      (frame_viz.shape[1], frame_viz.shape[0]),
                      (0, 0, 0),
                      -1)

        # 混合原始图像和背景
        alpha = 0.7
        cv2.addWeighted(help_bg, alpha, frame_viz, 1 - alpha, 0, frame_viz)

        # 添加关键快捷键提示
        hint_text = "Shortcuts: "
        key_list = ['q', 'd', 'r', 'h', 'm', 'l', 'w', 'p']
        for key in key_list:
            if key in self.key_hints:
                hint_text += f"'{key}'-{self.key_hints[key]} "

        cv2.putText(frame_viz, hint_text,
                    (10, frame_viz.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame_viz

    def draw_occlusion_message(self, frame):
        """
        绘制遮挡状态信息

        Args:
            frame: 输入帧

        Returns:
            ndarray: 带遮挡信息的帧
        """
        # 发布遮挡事件
        if self.publisher:
            self.publisher.publish_display_event(
                UIEventTypes.STATUS_CHANGED,
                status="occlusion"
            )

        frame_viz = frame.copy()

        # 使用半透明红色框作为遮挡指示
        overlay = frame_viz.copy()
        cv2.rectangle(overlay, (0, 0), (frame_viz.shape[1], frame_viz.shape[0]),
                      (0, 0, 200), -1)

        # 应用透明度
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame_viz, 1 - alpha, 0, frame_viz)

        # 添加文字提示
        cv2.putText(frame_viz, "Using Last Detection - Occlusion Status",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 添加通知
        self.add_notification("检测到遮挡，使用上次有效检测", "warning")

        return frame_viz

    def get_last_room_viz(self):
        """
        获取上次的房间视图

        Returns:
            ndarray: 上次的房间视图或新创建的空视图
        """
        if self.last_room_viz is None:
            self.last_room_viz = self.visualizer.visualize_room()
        return self.last_room_viz

    def toggle_advanced_info(self):
        """
        切换是否显示高级调试信息

        Returns:
            bool: 当前高级信息显示状态
        """
        # 保存旧状态
        old_state = self.show_advanced_info

        # 更新状态
        self.show_advanced_info = not old_state

        # 发布状态变更事件
        if self.publisher:
            self.publisher.publish_ui_state_change(
                UIEventTypes.OPTION_TOGGLED,
                "show_advanced_info",
                old_state,
                self.show_advanced_info
            )

        self.logger.info(
            f"高级调试信息显示: {'开启' if self.show_advanced_info else '关闭'}")
        return self.show_advanced_info

    def toggle_display_option(self, option_key):
        """
        切换显示选项

        Args:
            option_key: 选项键名

        Returns:
            bool: 选项当前状态
        """
        if option_key in self.display_options:
            # 保存旧状态
            old_value = self.display_options[option_key]

            # 更新状态
            self.display_options[option_key] = not old_value

            # 发布选项变更事件
            if self.publisher:
                self.publisher.publish_ui_state_change(
                    UIEventTypes.OPTION_TOGGLED,
                    option_key,
                    old_value,
                    self.display_options[option_key]
                )

            self.logger.info(
                f"显示选项 {option_key}: {'开启' if self.display_options[option_key] else '关闭'}")
            return self.display_options[option_key]
        return None

    def update_feature_state(self, feature_name, state):
        """
        更新功能开关状态
        Args:
            feature_name: 功能名称
            state: 功能状态
        Returns:
            bool: 更新是否成功
        """
        # 特殊处理异步状态
        if feature_name == 'async':
            # 如果状态已存在且没有变化，直接返回成功
            if feature_name in self.feature_states and self.feature_states[
                feature_name] == state:
                return True

            # 记录调用堆栈，但只在调试时详细记录
            import traceback
            short_stack = traceback.extract_stack()[-3:-1]  # 只取最近两个调用点
            stack_info = str(short_stack)
            self.logger.info(
                f"异步状态被更新为: {state}，调用来源: {stack_info}")

        # 常规状态更新逻辑
        if feature_name in self.feature_states:
            # 如果状态没有变化，不做任何事
            if self.feature_states[feature_name] == state:
                return True

            # 保存旧状态
            old_state = self.feature_states[feature_name]

            # 更新状态
            self.feature_states[feature_name] = state

            # 发布状态变更事件
            if self.publisher:
                self.publisher.publish_ui_state_change(
                    UIEventTypes.FEATURE_TOGGLED,
                    feature_name,
                    old_state,
                    state
                )

            self.logger.info(
                f"功能 '{feature_name}' 状态更新为: {'开启' if state else '关闭'}")

            # 如果更新了状态，确保UI刷新
            self.needs_redraw = True

            return True
        else:
            # 功能名称不存在于状态字典中，添加它
            self.feature_states[feature_name] = state
            self.logger.info(
                f"新增功能 '{feature_name}' 状态: {'开启' if state else '关闭'}")
            self.needs_redraw = True
            return True

    def setup_conditional_events(self, event_system):
        """
        设置条件性事件处理

        Args:
            event_system: 事件系统
        """
        # 保存事件系统引用
        self.event_system = event_system

        # 设置发布器和订阅器的事件系统
        self.publisher = UIEventPublisher.get_instance("display_manager")
        self.subscriber = UIEventSubscriber.get_instance("display_manager")

        # 设置条件性事件订阅
        from utils.conditional_events import get_conditional_event_system, where

        conditional_events = get_conditional_event_system(event_system)

        # 只当FPS低于15时显示警告
        conditional_events.subscribe_if(
            "fps_updated",
            where('fps').less_than(15),
            self.show_performance_warning
        )

        # 当检测状态变为"occlusion"时显示遮挡提示
        conditional_events.subscribe_if(
            "system_state_changed",
            where('state').equals('occlusion'),
            self.on_occlusion_state
        )

        # 设置基本事件处理器
        self._setup_event_handlers()

        self.logger.info("已设置条件性事件处理")

    def _setup_event_handlers(self):
        """设置基本事件处理器"""
        # 只有在有订阅器时才设置
        if not hasattr(self.subscriber, 'subscribe'):
            return

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

        self.logger.info("已设置基本事件处理器")

    def on_person_detected(self, data):
        """人体检测事件处理"""
        if 'person' in data:
            self.current_person = data['person']
            # 更新UI
            self._update_ui()

    def on_action_recognized(self, data):
        """动作识别事件处理"""
        if 'action' in data:
            self.current_action = data['action']
            # 更新UI
            self._update_ui()

    def on_position_mapped(self, data):
        """位置映射事件处理"""
        if 'position' in data:
            self.current_room_position = data['position']
            if 'depth' in data:
                self.current_depth = data['depth']

            # 更新轨迹点
            if self.visualizer and hasattr(self.visualizer, 'add_trail_point'):
                self.visualizer.add_trail_point(*self.current_room_position)

            # 更新UI
            self._update_ui()

    def on_feature_toggled(self, data):
        """功能开关事件处理"""
        if 'feature_name' in data and 'state' in data:
            feature_name = data['feature_name']
            state = data['state']

            # 直接更新状态，不触发事件，避免循环
            if feature_name in self.feature_states:
                self.feature_states[feature_name] = state

            # 更新UI
            self._update_ui()

    def on_system_state_changed(self, data):
        """系统状态变化事件处理"""
        if 'state' in data:
            self.current_system_state = data['state']
            # 更新UI
            self._update_ui()

    def on_fps_updated(self, data):
        """FPS更新事件处理"""
        if 'fps' in data:
            self.current_fps = data['fps']
            # 不立即更新UI，FPS信息会在下一次正常更新时显示

    def on_occlusion_state(self, data):
        """遮挡状态事件处理"""
        # 添加通知
        self.add_notification("检测到遮挡状态，使用上次的有效检测", "warning")
        # 更新UI
        self._update_ui()

    def show_performance_warning(self, data):
        """
        显示性能警告

        Args:
            data: 性能数据
        """
        if 'fps' in data:
            fps = data['fps']
            # 添加通知
            self.add_notification(f"性能警告: FPS={fps:.1f}", "warning")
            # 更新UI
            self._update_ui()

    def add_notification(self, message, level="info", duration=None):
        """
        添加通知

        Args:
            message: 通知消息
            level: 通知级别 (info, warning, error)
            duration: 显示时长(秒)，None使用默认值
        """
        if duration is None:
            duration = self.notification_duration

        # 添加到通知队列
        self.notifications.append(
            (message, level, duration, time.time() + duration)
        )

        # 发布通知事件
        if self.publisher:
            self.publisher.publish_notification(
                message,
                level=level,
                duration=duration
            )

        self.logger.info(f"添加通知: [{level}] {message}")

    def _update_ui(self):
        """更新UI（内部使用）"""
        current_time = time.time()

        # 限制更新频率
        if current_time - self.last_update_time < self.update_interval:
            return

        # 如果窗口尚未创建，先创建窗口
        if not self.windows_created:
            self.create_windows()

        # 更新相机显示（如果有当前帧）
        if self.current_frame is not None:
            camera_viz = self._create_camera_visualization()
            if camera_viz is not None:
                cv2.imshow('Camera View', camera_viz)

        # 更新房间视图（如果有当前位置）
        if self.current_room_position is not None:
            room_viz = self.visualizer.visualize_room(
                self.current_room_position,
                self.current_depth,
                self.current_action
            )
            if room_viz is not None:
                cv2.imshow('Room Position', room_viz)
                self.last_room_viz = room_viz.copy()

        # 更新时间戳
        self.last_update_time = current_time

        # 发布渲染完成事件
        if self.publisher:
            self.publisher.publish_display_event(
                UIEventTypes.RENDER_COMPLETED,
                timestamp=current_time
            )

    def _create_camera_visualization(self):
        """
        创建相机帧可视化（内部使用）

        Returns:
            ndarray: 处理后的相机帧
        """
        if self.current_frame is None:
            return None

        # 创建帧副本
        frame_viz = self.current_frame.copy()

        # 绘制人体检测结果
        if self.current_person and self.display_options["show_bbox"]:
            # 委托给可视化器绘制检测结果
            frame_viz = self.visualizer.visualize_frame(
                frame_viz,
                self.current_person,
                self.current_action
            )

        # 绘制调试信息
        if self.debug_mode:
            # 基本信息: FPS和状态
            if self.display_options["show_fps"]:
                cv2.putText(frame_viz, f"FPS: {self.current_fps:.1f}",
                            (frame_viz.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame_viz, f"Status: {self.current_system_state}",
                        (frame_viz.shape[1] - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 功能状态
            feature_y = 90
            for name, enabled in self.feature_states.items():
                color = (0, 255, 0) if enabled else (0, 0, 255)
                cv2.putText(frame_viz, f"{name}: {'ON' if enabled else 'OFF'}",
                            (frame_viz.shape[1] - 150, feature_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                feature_y += 25

            # 调试信息
            if self.show_advanced_info and self.current_debug_info:
                y_pos = feature_y + 10
                for key, value in self.current_debug_info.items():
                    text = f"{key}: {value}"
                    if isinstance(value, tuple) and len(value) == 2:
                        text = f"{key}: ({value[0]:.1f}, {value[1]:.1f})"
                    elif isinstance(value, (int, float)):
                        text = f"{key}: {value:.2f}"

                    cv2.putText(frame_viz, text,
                                (frame_viz.shape[1] - 250, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 25

            # 绘制通知
            self._draw_notifications(frame_viz)

            # 绘制快捷键提示
            self._draw_shortcut_hints(frame_viz)

        return frame_viz

    def _update_notifications(self):
        """更新通知队列，删除过期通知"""
        current_time = time.time()

        # 删除过期通知
        self.notifications = [
            n for n in self.notifications
            if n[3] > current_time  # 结束时间大于当前时间
        ]

    def _draw_notifications(self, frame):
        """
        绘制通知消息

        Args:
            frame: 输入帧
        """
        # 先更新通知队列，删除过期通知
        self._update_notifications()

        if not self.notifications:
            return

        y_offset = 120  # 通知起始Y位置

        for message, level, duration, end_time in self.notifications:
            # 确定颜色
            if level == "error":
                color = (0, 0, 255)  # 红色
            elif level == "warning":
                color = (0, 165, 255)  # 橙色
            else:
                color = (0, 255, 0)  # 绿色

            # 计算剩余时间比例
            remaining = end_time - time.time()
            if remaining <= 0:
                continue

            alpha = min(1.0, remaining / duration)

            # 绘制通知背景
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (10, y_offset - 20),
                (frame.shape[1] - 10, y_offset + 10),
                (50, 50, 50),
                -1
            )

            # 应用透明度
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # 绘制通知文本
            cv2.putText(
                frame,
                message,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1
            )

            y_offset += 40  # 为下一条通知留出空间

    def _draw_shortcut_hints(self, frame):
        """
        绘制快捷键提示

        Args:
            frame: 输入帧
        """
        # 绘制半透明背景
        help_bg_height = 30
        help_bg = frame.copy()
        cv2.rectangle(
            help_bg,
            (0, frame.shape[0] - help_bg_height),
            (frame.shape[1], frame.shape[0]),
            (0, 0, 0),
            -1
        )

        # 混合原始图像和背景
        alpha = 0.7
        cv2.addWeighted(help_bg, alpha, frame, 1 - alpha, 0, frame)

        # 添加快捷键提示
        hint_text = "快捷键: "
        key_list = ['q', 'd', 'r', 'h', 'm', 'l', 'w', 'p']
        for key in key_list:
            if key in self.key_hints:
                hint_text += f"'{key}'-{self.key_hints[key]} "

        cv2.putText(
            frame,
            hint_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    def cleanup(self):
        """
        清理资源

        在应用程序退出前调用，释放资源
        """
        try:
            # 关闭窗口
            if self.windows_created:
                cv2.destroyWindow('Camera View')
                cv2.destroyWindow('Room Position')

            # 取消事件订阅
            if hasattr(self.subscriber, 'unsubscribe_all'):
                self.subscriber.unsubscribe_all()
            if hasattr(self.publisher, 'cleanup'):
                self.publisher.cleanup()

            self.logger.info("显示管理器资源已清理")
            return True
        except Exception as e:
            self.logger.error(f"清理显示管理器资源时出错: {e}")
            return False
