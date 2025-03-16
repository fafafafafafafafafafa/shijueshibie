# -*- coding: utf-8 -*-
"""
Enhanced Visualizer Module - Provides visual feedback for the tracking system

This module provides:
1. Camera frame visualization with person detection overlay
2. Room position tracking visualization
3. Visual trails for movement tracking
4. Status display and debug information
5. Support for different view modes
"""
import cv2
import numpy as np
import time
import math
from collections import deque
import logging
from utils.logger_config import setup_logger
from utils.math_utils import MathUtils

# Configure logger
logger = setup_logger("EnhancedVisualizer")


class EnhancedVisualizer:
    """
    Enhanced visualization component for the tracking system

    Provides comprehensive visualization capabilities including:
    - Camera view with detection overlays
    - Room mapping visualization with position tracking
    - Movement trails and prediction visualization
    - Debug information display
    """

    def __init__(self, room_width=400, room_height=300, trail_length=50,
                 config=None):
        """
        Initialize the visualizer

        Args:
            room_width: Width of the room visualization in pixels
            room_height: Height of the room visualization in pixels
            trail_length: Maximum number of points in the movement trail
            config: Optional configuration dictionary
        """
        # Room visualization settings
        self.room_width = room_width
        self.room_height = room_height

        # Movement trail settings
        self.trail_length = trail_length
        self.trail_points = deque(maxlen=trail_length)
        self.trail_timestamps = deque(maxlen=trail_length)
        self.trail_colors = self._generate_trail_colors(trail_length)

        # Display settings
        self.show_skeleton = True
        self.show_bounding_box = True
        self.show_confidence = True
        self.show_action = True
        self.show_trail = True
        self.show_grid = True
        self.show_debug_info = True

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

        # Keypoint connection mappings for different detection models
        self.connections = {
            'mediapipe': [
                (0, 1), (1, 2), (2, 3), (3, 7),
                (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16), (0, 11), (0, 12)
            ],
            'default': [
                # Generic skeleton connections
                (0, 1), (1, 2), (2, 3), (3, 4),
                (1, 5), (5, 6), (6, 7), (1, 8),
                (8, 9), (9, 10), (1, 11), (11, 12),
                (12, 13)
            ]
        }

        # Apply custom configuration if provided
        self._apply_config(config)

        # Room visualization cache
        self.room_bg = self._create_room_background()

        # Last visualizations (for caching)
        self.last_frame_viz = None
        self.last_room_viz = None

        logger.info(
            f"EnhancedVisualizer initialized with room size: {room_width}x{room_height}, trail length: {trail_length}")

    def _apply_config(self, config):
        """Apply custom configuration settings"""
        if not config:
            return

        # Apply display settings from config
        if isinstance(config, dict):
            self.show_skeleton = config.get('show_skeleton', self.show_skeleton)
            self.show_bounding_box = config.get('show_bounding_box',
                                                self.show_bounding_box)
            self.show_confidence = config.get('show_confidence',
                                              self.show_confidence)
            self.show_action = config.get('show_action', self.show_action)
            self.show_trail = config.get('show_trail', self.show_trail)
            self.show_grid = config.get('show_grid', self.show_grid)
            self.show_debug_info = config.get('show_debug_info',
                                              self.show_debug_info)

            # Apply room dimensions from config
            self.room_width = config.get('room_width', self.room_width)
            self.room_height = config.get('room_height', self.room_height)

            # Apply trail settings from config
            if 'trail_length' in config:
                new_length = config['trail_length']
                if new_length != self.trail_length:
                    self.trail_length = new_length
                    self.trail_points = deque(maxlen=new_length)
                    self.trail_timestamps = deque(maxlen=new_length)
                    self.trail_colors = self._generate_trail_colors(new_length)

    def _create_room_background(self):
        """Create the static room background visualization"""
        # Create a blank room image
        room_bg = np.ones((self.room_height, self.room_width, 3),
                          dtype=np.uint8) * 255

        # Draw room border
        cv2.rectangle(room_bg, (0, 0),
                      (self.room_width - 1, self.room_height - 1), (0, 0, 0), 2)

        # Draw grid if enabled
        if self.show_grid:
            # Horizontal grid lines
            grid_spacing_y = self.room_height // 4
            for y in range(grid_spacing_y, self.room_height, grid_spacing_y):
                cv2.line(room_bg, (0, y), (self.room_width, y), (200, 200, 200),
                         1)

            # Vertical grid lines
            grid_spacing_x = self.room_width // 4
            for x in range(grid_spacing_x, self.room_width, grid_spacing_x):
                cv2.line(room_bg, (x, 0), (x, self.room_height),
                         (200, 200, 200), 1)

        # Add reference points and labels
        cv2.circle(room_bg, (self.room_width // 2, self.room_height // 2), 5,
                   (150, 150, 150), -1)  # Center point

        # Add cardinal directions
        cv2.putText(room_bg, "N", (self.room_width // 2, 15), self.font,
                    self.font_scale, (0, 0, 0), self.font_thickness)
        cv2.putText(room_bg, "S", (self.room_width // 2, self.room_height - 5),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)
        cv2.putText(room_bg, "W", (5, self.room_height // 2), self.font,
                    self.font_scale, (0, 0, 0), self.font_thickness)
        cv2.putText(room_bg, "E", (self.room_width - 15, self.room_height // 2),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        return room_bg

    def _generate_trail_colors(self, length):
        """Generate a gradient of colors for the trail visualization"""
        colors = []
        for i in range(length):
            # Transition from blue to red based on recency
            ratio = i / max(1, length - 1)
            b = int(255 * (1 - ratio))
            r = int(255 * ratio)
            g = 0
            colors.append((b, g, r))  # BGR format
        return colors

    def add_trail_point(self, x, y):
        """
        Add a point to the movement trail

        Args:
            x: X coordinate
            y: Y coordinate
        """
        if x is None or y is None:
            return

        # Add point to the trail
        self.trail_points.append((int(x), int(y)))
        self.trail_timestamps.append(time.time())

    def visualize_frame(self, frame, person=None, action=None, detector=None):
        """
        Visualize the camera frame with detection overlays
        Args:
            frame: Camera frame image
            person: Detected person data
            action: Recognized action
            detector: The detector instance (for model-specific visualizations)
        Returns:
            Image with detection visualizations
        """
        if frame is None:
            logger.warning("Cannot visualize: frame is None")
            return None

        # Make a copy of the frame to avoid modifying the original
        viz_frame = frame.copy()

        # Draw detection overlays if person data is available
        if person:
            # Draw keypoints and skeleton if available
            if 'keypoints' in person and self.show_skeleton:
                self._draw_skeleton(viz_frame, person, detector)

            # Draw bounding box if available
            if 'bbox' in person and self.show_bounding_box:
                self._draw_bounding_box(viz_frame, person['bbox'])

            # Draw confidence score if available
            if 'confidence' in person and self.show_confidence:
                conf_text = f"Confidence: {person['confidence']:.2f}"
                cv2.putText(viz_frame, conf_text, (10, 20), self.font,
                            self.font_scale, (0, 255, 0), self.font_thickness)

        # Draw action label if available
        if action and self.show_action:
            action_text = f"Action: {action}"
            cv2.putText(viz_frame, action_text, (10, 40), self.font,
                        self.font_scale, (0, 255, 0), self.font_thickness)

        # Cache the visualization
        self.last_frame_viz = viz_frame
        return viz_frame

    def _draw_skeleton(self, frame, person, detector=None):
        """Draw the pose skeleton on the frame"""
        keypoints = person.get('keypoints', [])

        # 修改这一行 - 正确处理NumPy数组
        if isinstance(keypoints, np.ndarray):
            # 对于NumPy数组，检查是否为空或长度为0
            if keypoints.size == 0:
                return
        else:
            # 对于列表等其他类型，使用标准判断
            if not keypoints:
                return

        # 验证关键点格式 - 需要适配NumPy数组
        try:
            if isinstance(keypoints, np.ndarray):
                if keypoints.ndim < 2 or keypoints.shape[1] < 3:
                    return
            else:
                if len(keypoints) > 0:
                    first_kp = keypoints[0]
                    if not isinstance(first_kp, (list, tuple)) or len(
                            first_kp) < 3:
                        return
        except Exception:
            return

        # Determine which connection map to use
        connection_style = 'default'
        if detector and hasattr(detector,
                                'using_mediapipe') and detector.using_mediapipe:
            connection_style = 'mediapipe'

        connections = self.connections.get(connection_style,
                                           self.connections['default'])

        # Draw keypoints - 需要适配NumPy数组
        try:
            if isinstance(keypoints, np.ndarray):
                # 对于NumPy数组，使用向量化操作
                indices = np.where(keypoints[:, 2] > 0.2)[0]
                for i in indices:
                    x, y, conf = keypoints[i]
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            else:
                # 对于列表，使用循环
                for i, kp in enumerate(keypoints):
                    if len(kp) < 3:
                        continue
                    x, y, conf = kp
                    if conf > 0.05:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        except Exception:
            pass

        # Draw skeleton connections - 需要适配NumPy数组
        for connection in connections:
            try:
                idx1, idx2 = connection

                if isinstance(keypoints, np.ndarray):
                    if idx1 < keypoints.shape[0] and idx2 < keypoints.shape[0]:
                        x1, y1, conf1 = keypoints[idx1]
                        x2, y2, conf2 = keypoints[idx2]

                        if conf1 > 0.05 and conf2 > 0.05:
                            cv2.line(frame, (int(x1), int(y1)),
                                     (int(x2), int(y2)), (0, 255, 255), 2)
                else:
                    if idx1 < len(keypoints) and idx2 < len(keypoints):
                        kp1 = keypoints[idx1]
                        kp2 = keypoints[idx2]

                        if len(kp1) < 3 or len(kp2) < 3:
                            continue

                        x1, y1, conf1 = kp1
                        x2, y2, conf2 = kp2
                        if conf1 > 0.2 and conf2 > 0.2:
                            cv2.line(frame, (int(x1), int(y1)),
                                     (int(x2), int(y2)), (0, 255, 255), 2)
            except Exception:
                pass

    def _draw_bounding_box(self, frame, bbox):
        """Draw a bounding box on the frame"""
        try:
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)),
                              (255, 0, 0), 2)
        except Exception as e:
            logger.error(f"Error drawing bounding box: {e}")

    def _draw_trail(self, room_viz):
        """Draw the movement trail on the room visualization"""
        if len(self.trail_points) < 2:
            return

        # Draw lines connecting trail points with gradient coloring
        for i in range(1, len(self.trail_points)):
            start_point = self.trail_points[i - 1]
            end_point = self.trail_points[i]
            color = self.trail_colors[i] if i < len(self.trail_colors) else (
            0, 0, 255)
            cv2.line(room_viz, start_point, end_point, color, 2)

        # Draw small circles at each trail point
        for i, point in enumerate(self.trail_points):
            color = self.trail_colors[i] if i < len(self.trail_colors) else (
            0, 0, 255)
            cv2.circle(room_viz, point, 3, color, -1)

        # Draw the most recent point with a larger radius
        if self.trail_points:
            latest_point = self.trail_points[-1]
            cv2.circle(room_viz, latest_point, 5, (0, 0, 255), -1)

    def calculate_movement_metrics(self):
        """
        Calculate movement metrics based on the trail data

        Returns:
            dict: Movement metrics including speed, distance, etc.
        """
        metrics = {
            'speed': 0.0,  # pixels per second
            'total_distance': 0.0,  # pixels
            'smoothness': 0.0,  # ratio of direct distance to total path (0-1)
            'active_time': 0.0  # seconds with movement detected
        }

        if len(self.trail_points) < 2 or len(self.trail_timestamps) < 2:
            return metrics

        # Calculate total distance and point-to-point distances
        total_distance = 0.0
        distances = []

        for i in range(1, len(self.trail_points)):
            p1 = self.trail_points[i - 1]
            p2 = self.trail_points[i]
            dist = MathUtils.distance(p1, p2)
            total_distance += dist
            distances.append(dist)

        metrics['total_distance'] = total_distance

        # Calculate speed (using most recent points)
        if len(self.trail_points) >= 2 and len(self.trail_timestamps) >= 2:
            recent_time = self.trail_timestamps[-1] - self.trail_timestamps[
                -min(10, len(self.trail_timestamps))]
            recent_dist = sum(distances[-min(9, len(distances)):])
            if recent_time > 0:
                metrics['speed'] = recent_dist / recent_time

        # Calculate smoothness (ratio of direct distance to total path)
        if total_distance > 0 and len(self.trail_points) >= 2:
            direct_distance = MathUtils.distance(self.trail_points[0],
                                                 self.trail_points[-1])
            metrics['smoothness'] = direct_distance / total_distance

        # Calculate active time (time between first and last timestamp with movement)
        if len(self.trail_timestamps) >= 2:
            metrics['active_time'] = self.trail_timestamps[-1] - \
                                     self.trail_timestamps[0]

        return metrics

    def draw_debug_info(self, frame, fps=None, system_state=None,
                        debug_data=None):
        """
        Draw debug information on the frame

        Args:
            frame: Input frame
            fps: Current FPS
            system_state: Current system state
            debug_data: Additional debug data to display

        Returns:
            Frame with debug information
        """
        if not self.show_debug_info or frame is None:
            return frame

        # Create a copy of the frame
        debug_frame = frame.copy()

        # Draw semi-transparent background for debug info
        overlay = debug_frame.copy()
        cv2.rectangle(overlay, (0, 0), (200, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, debug_frame, 0.5, 0, debug_frame)

        # Draw FPS
        y_pos = 20
        if fps is not None:
            cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, y_pos),
                        self.font, self.font_scale, (255, 255, 255),
                        self.font_thickness)
            y_pos += 20

        # Draw system state
        if system_state:
            cv2.putText(debug_frame, f"State: {system_state}", (10, y_pos),
                        self.font, self.font_scale, (255, 255, 255),
                        self.font_thickness)
            y_pos += 20

        # Draw movement metrics
        metrics = self.calculate_movement_metrics()
        if metrics['speed'] > 0:
            cv2.putText(debug_frame, f"Speed: {metrics['speed']:.1f} px/s",
                        (10, y_pos),
                        self.font, self.font_scale, (255, 255, 255),
                        self.font_thickness)
            y_pos += 20

        if metrics['total_distance'] > 0:
            cv2.putText(debug_frame,
                        f"Distance: {metrics['total_distance']:.1f} px",
                        (10, y_pos),
                        self.font, self.font_scale, (255, 255, 255),
                        self.font_thickness)
            y_pos += 20

        # Draw additional debug data
        if debug_data and isinstance(debug_data, dict):
            for key, value in debug_data.items():
                if y_pos < frame.shape[
                    0] - 10:  # Ensure we don't draw outside frame
                    cv2.putText(debug_frame, f"{key}: {value}", (10, y_pos),
                                self.font, self.font_scale, (255, 255, 255),
                                self.font_thickness)
                    y_pos += 20

        return debug_frame

    def draw_feature_states(self, frame, features=None):
        """
        Draw feature states on the frame

        Args:
            frame: Input frame
            features: Dictionary of feature states {feature_name: is_enabled}

        Returns:
            Frame with feature states information
        """
        if frame is None or not features:
            return frame

        # Create a copy of the frame
        feature_frame = frame.copy()

        # Draw semi-transparent background in top-right
        width = frame.shape[1]
        overlay = feature_frame.copy()
        cv2.rectangle(overlay, (width - 180, 0), (width, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, feature_frame, 0.5, 0, feature_frame)

        # Draw feature states
        y_pos = 20
        cv2.putText(feature_frame, "Features:", (width - 170, y_pos),
                    self.font, self.font_scale, (255, 255, 255),
                    self.font_thickness)
        y_pos += 20

        for feature, enabled in features.items():
            color = (0, 255, 0) if enabled else (0, 0, 255)
            status = "ON" if enabled else "OFF"
            cv2.putText(feature_frame, f"{feature}: {status}",
                        (width - 170, y_pos),
                        self.font, self.font_scale, color, self.font_thickness)
            y_pos += 20

        return feature_frame

    def draw_detection_visualization(self, frame, detection_result):
        """
        Draw comprehensive detection visualization

        Args:
            frame: Input frame
            detection_result: Complete detection result from the pipeline

        Returns:
            Frame with comprehensive visualization
        """
        if frame is None or not detection_result:
            return frame

        # Extract data from detection result
        person = detection_result.get('person')
        action = detection_result.get('action')
        position = detection_result.get('position')

        # Create base visualization
        viz_frame = self.visualize_frame(frame, person, action)

        # Draw additional information
        if position:
            position_text = f"Position: ({position[0]:.1f}, {position[1]:.1f})"
            cv2.putText(viz_frame, position_text, (10, 60), self.font,
                        self.font_scale, (0, 255, 0), self.font_thickness)

        return viz_frame

    def draw_occlusion_message(self, frame):
        """
        Draw occlusion message when person is temporarily occluded

        Args:
            frame: Input frame

        Returns:
            Frame with occlusion message
        """
        if frame is None:
            return None

        # Create a copy of the frame
        occlusion_frame = frame.copy()

        # Draw semi-transparent overlay
        overlay = occlusion_frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, occlusion_frame, 0.7, 0, occlusion_frame)

        # Draw occlusion message
        message = "Person Occluded"
        font_scale = 1.0
        thickness = 2
        text_size = cv2.getTextSize(message, self.font, font_scale, thickness)[
            0]
        text_x = int((w - text_size[0]) / 2)
        text_y = int((h + text_size[1]) / 2)

        # Draw text with border for better visibility
        cv2.putText(occlusion_frame, message, (text_x, text_y), self.font,
                    font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(occlusion_frame, message, (text_x, text_y), self.font,
                    font_scale, (255, 255, 255), thickness)

        # Draw secondary message
        secondary = "Prediction active..."
        sec_scale = 0.7
        sec_size = cv2.getTextSize(secondary, self.font, sec_scale, 1)[0]
        sec_x = int((w - sec_size[0]) / 2)
        sec_y = text_y + 40

        cv2.putText(occlusion_frame, secondary, (sec_x, sec_y), self.font,
                    sec_scale, (0, 0, 0), 3)
        cv2.putText(occlusion_frame, secondary, (sec_x, sec_y), self.font,
                    sec_scale, (100, 200, 255), 1)

        return occlusion_frame

    def draw_welcome_screen(self, width=640, height=480):
        """
        Draw welcome screen when application starts

        Args:
            width: Screen width
            height: Screen height

        Returns:
            Welcome screen image
        """
        # Create blank image
        welcome = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw background gradient
        for y in range(height):
            color_value = int(220 * (1 - y / height))
            cv2.line(welcome, (0, y), (width, y),
                     (color_value, color_value, color_value), 1)

        # Draw title
        title = "Tracking System"
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(title, self.font, font_scale, thickness)[0]
        text_x = int((width - text_size[0]) / 2)
        text_y = int(height * 0.3)

        cv2.putText(welcome, title, (text_x, text_y), self.font, font_scale,
                    (0, 100, 200), thickness)

        # Draw instructions
        instructions = [
            "Press 'r' to recalibrate",
            "Press 'h' for help",
            "Press 'q' to quit",
            "Press 'y' to toggle async mode"
        ]

        y_pos = int(height * 0.5)
        for instruction in instructions:
            text_size = cv2.getTextSize(instruction, self.font, 0.7, 1)[0]
            text_x = int((width - text_size[0]) / 2)
            cv2.putText(welcome, instruction, (text_x, y_pos), self.font, 0.7,
                        (0, 0, 0), 1)
            y_pos += 30

        # Draw version info
        version = "Version 1.0"
        text_size = cv2.getTextSize(version, self.font, 0.6, 1)[0]
        cv2.putText(welcome, version, (width - text_size[0] - 10, height - 20),
                    self.font, 0.6, (100, 100, 100), 1)

        return welcome

    def get_debug_info(self):
        """
        Get debug information about the visualizer

        Returns:
            dict: Debug information
        """
        metrics = self.calculate_movement_metrics()

        debug_info = {
            'trail_length': len(self.trail_points),
            'movement_speed': f"{metrics['speed']:.2f} px/s",
            'total_distance': f"{metrics['total_distance']:.2f} px",
            'smoothness': f"{metrics['smoothness']:.2f}",
            'visualizer_mode': "Normal"
        }

        return debug_info

    def visualize_room(self, position=None, depth=None, action=None):
        """
        Visualize room map with person position

        Args:
            position: (x, y) position in room coordinates
            depth: Depth/distance estimation
            action: Recognized action

        Returns:
            Room visualization image
        """
        print("======= 房间可视化开始 =======")

        # Start with a copy of the room background
        room_viz = self.room_bg.copy()

        # Draw the movement trail if enabled
        if self.show_trail and len(self.trail_points) > 1:
            print(f"绘制运动轨迹: {len(self.trail_points)}个点")
            self._draw_trail(room_viz)
        else:
            print(
                f"轨迹显示: {self.show_trail}, 点数: {len(self.trail_points)}")

        # Draw the current position if available
        if position:
            x, y = position
            print(f"原始位置: x={x}, y={y}")

            # Ensure position is within room boundaries
            x = max(0, min(x, self.room_width))
            y = max(0, min(y, self.room_height))
            if (x, y) != position:
                print(f"位置已调整至边界内: x={x}, y={y}")

            # Draw different markers based on action
            if action and action != "Static":
                # Active state - larger filled circle
                cv2.circle(room_viz, (int(x), int(y)), 10, (0, 0, 255), -1)
                print(f"绘制活动状态标记: action={action}")

                # Draw action label
                label_pos = (int(x) + 15, int(y) - 5)
                cv2.putText(room_viz, action, label_pos, self.font,
                            self.font_scale, (0, 0, 255), self.font_thickness)
            else:
                # Static state - smaller circle with thick border
                cv2.circle(room_viz, (int(x), int(y)), 8, (0, 128, 255), 3)
                print(f"绘制静止状态标记: action={action}")

            # Visualize depth if available
            if depth is not None:
                # Draw a semi-transparent circle to represent depth
                depth_radius = min(100, max(10, int(depth * 10)))
                overlay = room_viz.copy()
                cv2.circle(overlay, (int(x), int(y)), depth_radius,
                           (100, 100, 255), -1)
                # Apply transparency
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, room_viz, 1 - alpha, 0,
                                room_viz)

                # Draw depth value
                depth_text = f"Depth: {depth:.1f}m"
                cv2.putText(room_viz, depth_text, (10, self.room_height - 10),
                            self.font, self.font_scale, (0, 0, 0),
                            self.font_thickness)
                print(f"可视化深度: {depth:.2f}m, 半径: {depth_radius}")
        else:
            print("警告: 没有位置数据提供")

        # Cache the visualization
        self.last_room_viz = room_viz
        print("======= 房间可视化结束 =======")

        return room_viz
