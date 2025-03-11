# -*- coding: utf-8 -*-
import cv2
import logging
import numpy as np
from utils.logger_config import setup_logger, init_root_logger, setup_utf8_console
from utils.conditional_events import get_conditional_event_system, where

logger = setup_logger(__name__)  # 使用模块名作为日志记录器名称


class DisplayManager:
    """负责处理显示和用户界面，支持多种显示模式和调试信息"""

    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.last_room_viz = None
        self.debug_mode = True
        self.show_advanced_info = False
        self.windows_created = False
        self.logger = logger
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
            'q': 'exit',
            'd': 'Debug',
            'r': 'recal',
            'f': 'FPS display',
            's': 'Skeleton display',
            'b': 'frame display',
            't': 'track display',
            'a': 'Motion Display',
            'h': 'Help',
            'm': 'MediaPipe',
            'l': 'ML Model',
            'w': 'DTW',
            'p': 'Threading',
            'y': 'Async'
        }

    def create_windows(self):
        """创建并配置显示窗口"""
        if not self.windows_created:
            cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera View', 800, 600)
            cv2.namedWindow('Room Position', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Room Position', 800, 600)
            self.windows_created = True
            self.logger.info("创建显示窗口")

    def draw_debug_info(self, frame, fps, system_state, debug_info=None):
        """绘制调试信息

        Args:
            frame: 输入帧
            fps: 当前帧率
            system_state: 系统状态
            debug_info: 附加调试信息

        Returns:
            ndarray: 带调试信息的帧
        """
        if not self.debug_mode:
            return frame

        frame_viz = frame.copy()

        # 基本信息: FPS和状态
        if self.display_options["show_fps"]:
            cv2.putText(frame_viz, f"FPS: {fps:.1f}",
                        (frame_viz.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame_viz, f"Status: {system_state}",
                    (frame_viz.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 添加功能开关状态 - 始终显示所有功能状态，不再依赖self.show_advanced_info
        feature_y = 90
        for name, enabled in self.feature_states.items():
            color = (0, 255, 0) if enabled else (0, 0, 255)  # 绿色为开启，红色为关闭
            cv2.putText(frame_viz, f"{name}: {'ON' if enabled else 'OFF'}",
                        (frame_viz.shape[1] - 150, feature_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            feature_y += 25

        # 添加功能开关状态
        if self.show_advanced_info:
            # 显示功能开关状态
            feature_y = 90
            for name, enabled in self.feature_states.items():
                color = (0, 255, 0) if enabled else (0, 0, 255)  # 绿色为开启，红色为关闭
                cv2.putText(frame_viz, f"{name}: {'ON' if enabled else 'OFF'}",
                            (frame_viz.shape[1] - 150, feature_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                feature_y += 25
            # 如果启用了高级信息显示，继续显示其他调试信息
            if self.show_advanced_info:
                # 高级调试信息
                if debug_info:
                    y_pos = feature_y + 10  # 在功能状态之后显示其他调试信息
                    for key, value in debug_info.items():
                        text = f"{key}: {value}"
                        if isinstance(value, tuple) and len(value) == 2:
                            text = f"{key}: ({value[0]:.1f}, {value[1]:.1f})"
                        elif isinstance(value, (int, float)):
                            text = f"{key}: {value:.2f}"

                        cv2.putText(frame_viz, text,
                                    (frame_viz.shape[1] - 250, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
                        y_pos += 25

                        # 高级调试信息
            if debug_info:
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
        elif debug_info:
            # 如果没有显示高级信息，但有调试信息，仍显示基本调试信息
            y_pos = 90
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

        # 绘制快捷键提示 (更改为更清晰的格式)
        # 使用半透明的黑色条作为背景，提高可读性
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
            hint_text += f"'{key}'-{self.key_hints[key]} "

        cv2.putText(frame_viz, hint_text,
                    (10, frame_viz.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame_viz

    def draw_occlusion_message(self, frame):
        """绘制遮挡状态信息

        Args:
            frame: 输入帧

        Returns:
            ndarray: 带遮挡信息的帧
        """
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
        return frame_viz

    def display_frames(self, camera_view, room_view):
        """显示相机视图和房间视图

        Args:
            camera_view: 相机视图帧
            room_view: 房间平面图视图
        """
        if not self.windows_created:
            self.create_windows()

        try:
            if camera_view is None:
                self.logger.warning("Camera view is None, skip display")
                return

            if not isinstance(camera_view, np.ndarray):
                self.logger.warning(
                    f"相机视图类型错误: {type(camera_view)}，跳过显示")
                return

            self.logger.info(f"相机视图尺寸: {camera_view.shape}")

            if camera_view.size == 0 or camera_view.shape[0] <= 0 or \
                    camera_view.shape[1] <= 0:
                self.logger.warning("相机视图尺寸无效，跳过显示")
                return

            if room_view is None:
                self.logger.warning("房间视图为None，尝试使用上次的有效视图")
                room_view = self.get_last_room_viz()

            if room_view is not None:
                self.logger.info(f"房间视图尺寸: {room_view.shape}")

            if room_view is None or room_view.size == 0 or room_view.shape[
                0] <= 0 or room_view.shape[1] <= 0:
                self.logger.warning("房间视图无效，跳过显示")
                return

            # 显示图像
            cv2.imshow('Camera View', camera_view)
            cv2.imshow('Room Position', room_view)
            self.last_room_viz = room_view.copy()  # 保存有效副本

        except Exception as e:
            self.logger.error(f"显示帧错误: {e}")

    def get_last_room_viz(self):
        """获取上次的房间视图

        Returns:
            ndarray: 上次的房间视图或新创建的空视图
        """
        if self.last_room_viz is None:
            self.last_room_viz = self.visualizer.visualize_room()
        return self.last_room_viz

    def toggle_advanced_info(self):
        """切换是否显示高级调试信息

        Returns:
            bool: 当前高级信息显示状态
        """
        self.show_advanced_info = not self.show_advanced_info
        self.logger.info(
            f"高级调试信息显示: {'开启' if self.show_advanced_info else '关闭'}")
        return self.show_advanced_info

    def toggle_display_option(self, option_key):
        """切换显示选项

        Args:
            option_key: 选项键名

        Returns:
            bool: 选项当前状态
        """
        if option_key in self.display_options:
            self.display_options[option_key] = not self.display_options[
                option_key]
            self.logger.info(
                f"显示选项 {option_key}: {'开启' if self.display_options[option_key] else '关闭'}")
            return self.display_options[option_key]
        return None

    def update_feature_state(self, feature_name, state):
        """更新功能开关状态

        Args:
            feature_name: 功能名称
            state: 功能状态

        Returns:
            bool: 更新是否成功
        """
        if feature_name in self.feature_states:
            self.feature_states[feature_name] = state
            self.logger.info(
                f"功能 '{feature_name}' 状态更新为: {'开启' if state else '关闭'}")
            return True
        return False

    def setup_conditional_events(self, event_system):
        """设置条件性事件处理"""
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
            self.show_occlusion_message
        )

    def show_performance_warning(self, data):
        """显示性能警告"""
        fps = data.get('fps', 0)
        cv2.putText(self.current_frame, f"性能警告: FPS={fps:.1f}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


