def add_trail_point(self, x, y, z=None):
    """
    添加轨迹点

    Args:
        x: X坐标
        y: Y坐标
        z: Z坐标（可选）
    """
    if not self._trail_enabled:
        return

    # 如果没有提供Z坐标，使用默认值
    if z is None:
        z = 0

    # 添加到轨迹
    self._trail_points.append((x, y, z))

    # 限制轨迹长度
    if len(self._trail_points) > self._trail_length:
        self._trail_points = self._trail_points[-self._trail_length:]


def draw_debug_info(self, frame, fps=None, system_state=None, debug_data=None):
    """
    绘制调试信息

    Args:
        frame: 输入帧
        fps: 当前帧率
        system_state: 系统状态
        debug_data: 调试数据

    Returns:
        ndarray: 带调试信息的帧
    """
    if not self.is_enabled() or not self._show_debug_info:
        return frame

    try:
        # 创建帧副本
        debug_frame = frame.copy()

        # 添加半透明背景
        overlay = debug_frame.copy()
        cv2.rectangle(overlay, (0, 0), (200, 130), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, debug_frame, 1 - alpha, 0, debug_frame)

        # 添加基本信息
        y_pos = 30

        # 帧率
        if fps is not None:
            cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25

        # 系统状态
        if system_state:
            cv2.putText(debug_frame, f"State: {system_state}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25

        # 添加额外调试数据
        if debug_data:
            for key, value in debug_data.items():
                # 限制显示数量
                if y_pos > frame.shape[0] - 30:
                    break

                text = f"{key}: {value}"
                cv2.putText(debug_frame, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20

        return debug_frame

    except Exception as e:
        logger.error(f"绘制调试信息时出错: {e}")
        return frame


def draw_occlusion_message(self, frame):
    """
    绘制遮挡信息

    Args:
        frame: 输入帧

    Returns:
        ndarray: 带遮挡信息的帧
    """
    if not self.is_enabled():
        return frame

    try:
        # 创建帧副本
        occlusion_frame = frame.copy()

        # 添加半透明红色遮罩
        overlay = occlusion_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 200), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, occlusion_frame, 1 - alpha, 0,
                        occlusion_frame)

        # 添加文本
        text = "Occlusion Detected"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2

        cv2.putText(occlusion_frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 如果有最后的3D视图，添加到角落
        if self._last_keypoints_3d is not None:
            occlusion_frame = self._add_3d_view(occlusion_frame,
                                                self._last_keypoints_3d,
                                                "Unknown")

        return occlusion_frame

    except Exception as e:
        logger.error(f"绘制遮挡信息时出错: {e}")
        return frame


def get_debug_info(self):
    """
    获取调试信息

    Returns:
        dict: 调试信息
    """
    return {
        'visualizer_type': '3d',
        'trail_points': len(self._trail_points),
        'view_angle': self._view_angle,
        'rotation_angle': self._rotation_angle,
        'elevation_angle': self._elevation_angle,
        'color_scheme': self._color_scheme,
        'enabled': self._enabled
    }


# ============= 辅助方法 =============

def _setup_colors(self):
    """
    设置颜色方案

    Returns:
        dict: 颜色设置
    """
    if self._color_scheme == 'default':
        return {
            'background': (0, 0, 0),  # 黑色背景
            'grid': (50, 50, 50),  # 深灰色网格
            'axes': [(0, 0, 255), (0, 255, 0), (255, 0, 0)],  # 蓝-绿-红 (x-y-z)
            'skeleton': (0, 255, 255),  # 青色骨架
            'joints': (255, 255, 0),  # 黄色关节
            'position': (0, 255, 0),  # 绿色位置标记
            'trail': [(0, 0, 255), (255, 0, 0)],  # 蓝到红色渐变
            'floor': (80, 80, 80),  # 灰色地面
            'text': (255, 255, 255)  # 白色文本
        }
    elif self._color_scheme == 'light':
        return {
            'background': (240, 240, 240),  # 浅灰色背景
            'grid': (180, 180, 180),  # 中灰色网格
            'axes': [(255, 0, 0), (0, 150, 0), (0, 0, 255)],  # 红-绿-蓝 (x-y-z)
            'skeleton': (70, 130, 180),  # 钢青色骨架
            'joints': (220, 20, 60),  # 深红色关节
            'position': (50, 205, 50),  # 青柠色位置标记
            'trail': [(65, 105, 225), (220, 20, 60)],  # 蓝到红色渐变
            'floor': (200, 200, 200),  # 浅灰色地面
            'text': (0, 0, 0)  # 黑色文本
        }
    elif self._color_scheme == 'dark':
        return {
            'background': (30, 30, 30),  # 深灰色背景
            'grid': (70, 70, 70),  # 中灰色网格
            'axes': [(0, 120, 255), (0, 200, 0), (200, 0, 0)],  # 蓝-绿-红 (x-y-z)
            'skeleton': (0, 200, 200),  # 青色骨架
            'joints': (200, 200, 0),  # 黄色关节
            'position': (0, 200, 0),  # 绿色位置标记
            'trail': [(0, 120, 255), (200, 0, 0)],  # 蓝到红色渐变
            'floor': (50, 50, 50),  # 深灰色地面
            'text': (220, 220, 220)  # 浅灰色文本
        }
    else:
        # 默认颜色
        return self._setup_colors('default')


def _setup_skeleton(self):
    """
    设置骨架连接

    Returns:
        list: 骨架连接列表
    """
    # 定义COCO 17关键点格式的连接
    return [
        (0, 1), (0, 2),  # 鼻子到左右眼
        (1, 3), (2, 4),  # 左右眼到左右耳
        (0, 5), (0, 6),  # 鼻子到左右肩
        (5, 7), (7, 9),  # 左肩到左肘到左手腕
        (6, 8), (8, 10),  # 右肩到右肘到右手腕
        (5, 11), (6, 12),  # 左肩到左髋，右肩到右髋
        (11, 13), (13, 15),  # 左髋到左膝到左踝
        (12, 14), (14, 16),  # 右髋到右膝到右踝
        (11, 12)  # 左髋到右髋
    ]


def _create_room_bg(self):
    """
    创建房间背景图
    """
    # 创建空白图像
    room_bg = np.ones((self._room_height, self._room_width, 3),
                      dtype=np.uint8) * 255

    # 绘制边框
    cv2.rectangle(room_bg, (0, 0),
                  (self._room_width - 1, self._room_height - 1), (0, 0, 0), 2)

    # 绘制网格
    if self._show_grid:
        # 水平线
        grid_h_spacing = self._room_height // 5
        for i in range(1, 5):
            y = i * grid_h_spacing
            cv2.line(room_bg, (0, y), (self._room_width, y), (200, 200, 200), 1)

        # 垂直线
        grid_w_spacing = self._room_width // 5
        for i in range(1, 5):
            x = i * grid_w_spacing
            cv2.line(room_bg, (x, 0), (x, self._room_height), (200, 200, 200),
                     1)

    # 绘制坐标轴
    if self._show_axes:
        # X轴
        cv2.arrowedLine(room_bg,
                        (10, self._room_height - 10),
                        (100, self._room_height - 10),
                        (0, 0, 200), 2)
        cv2.putText(room_bg, "X", (105, self._room_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        # Y轴
        cv2.arrowedLine(room_bg,
                        (10, self._room_height - 10),
                        (10, self._room_height - 100),
                        (0, 200, 0), 2)
        cv2.putText(room_bg, "Y", (5, self._room_height - 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    # 保存背景图
    self._room_bg = room_bg


def _draw_2d_skeleton(self, frame, keypoints, action=None):
    """
    在图像上绘制2D骨架

    Args:
        frame: 输入帧
        keypoints: 关键点列表
        action: 当前动作

    Returns:
        ndarray: 带骨架的帧
    """
    # 检查关键点数据是否足够
    if not keypoints or len(keypoints) < 5:
        return frame

    # 绘制关节点和连接
    for i, kp in enumerate(keypoints):
        if len(kp) >= 3 and kp[2] > 0.2:  # 只绘制置信度高于0.2的关键点
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    # 绘制骨架连接
    for conn in self._skeleton_connections:
        i, j = conn
        if i < len(keypoints) and j < len(keypoints):
            if len(keypoints[i]) >= 3 and len(keypoints[j]) >= 3:
                if keypoints[i][2] > 0.2 and keypoints[j][2] > 0.2:
                    pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                    pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    return frame


def _estimate_3d_keypoints(self, keypoints_2d):
    """
    从2D关键点估计3D关键点

    Args:
        keypoints_2d: 2D关键点列表

    Returns:
        list: 3D关键点列表
    """
    # 这是一个简化版本，实际应用中可能需要更复杂的3D姿态估计算法
    keypoints_3d = []

    # 确保有足够的关键点
    if not keypoints_2d or len(keypoints_2d) < 5:
        return None

    # 计算深度比例因子（简化版）
    # 如果有肩膀关键点，使用肩宽
    shoulder_width = 0
    if len(keypoints_2d) > 6:
        left_shoulder = keypoints_2d[5]
        right_shoulder = keypoints_2d[6]
        if len(left_shoulder) >= 3 and len(right_shoulder) >= 3:
            if left_shoulder[2] > 0.2 and right_shoulder[2] > 0.2:
                shoulder_width = math.sqrt(
                    (left_shoulder[0] - right_shoulder[0]) ** 2 +
                    (left_shoulder[1] - right_shoulder[1]) ** 2)

    depth_scale = 100.0 / (shoulder_width if shoulder_width > 0 else 100.0)

    # 为每个关键点估计Z坐标
    for kp in keypoints_2d:
        if len(kp) >= 3:
            x, y, conf = kp
            # 使用Y坐标计算伪深度（越靠上的点越远）
            normalized_y = (
                                       frame_height - y) / frame_height if 'frame_height' in locals() else 0.5
            z = depth_scale * normalized_y * 100

            keypoints_3d.append([x, y, z, conf])
        else:
            keypoints_3d.append([0, 0, 0, 0])  # 无效关键点

    return keypoints_3d


def _add_3d_view(self, frame, keypoints_3d, action=None):
    """
    在图像上添加3D视图

    Args:
        frame: 输入帧
        keypoints_3d: 3D关键点列表
        action: 当前动作

    Returns:
        ndarray: 带3D视图的帧
    """
    # 设置3D视图区域大小和位置
    view_size = 200
    margin = 20
    view_x = frame.shape[1] - view_size - margin
    view_y = margin

    # 创建3D视图
    view_bg = np.zeros((view_size, view_size, 3), dtype=np.uint8)
    view_bg[:] = self._colors['background']

    # 计算3D变换（简化版）
    rotation_rad = math.radians(self._rotation_angle)
    elevation_rad = math.radians(self._elevation_angle)

    # 投影3D关键点到2D
    points_2d = []
    for kp in keypoints_3d:
        if len(kp) >= 4 and kp[3] > 0.2:  # 使用置信度
            x, y, z = kp[0], kp[1], kp[2]

            # 旋转和投影
            x_rot = x * math.cos(rotation_rad) - z * math.sin(rotation_rad)
            z_rot = x * math.sin(rotation_rad) + z * math.cos(rotation_rad)

            y_rot = y * math.cos(elevation_rad) - z_rot * math.sin(
                elevation_rad)
            z_rot = y * math.sin(elevation_rad) + z_rot * math.cos(
                elevation_rad)

            # 缩放和居中
            scale = self._zoom_factor * view_size / 300.0
            px = int(view_size / 2 + x_rot * scale)
            py = int(view_size / 2 + y_rot * scale)

            points_2d.append((px, py))
        else:
            points_2d.append(None)  # 无效点

    # 绘制骨架
    for conn in self._skeleton_connections:
        i, j = conn
        if i < len(points_2d) and j < len(points_2d):
            if points_2d[i] is not None and points_2d[j] is not None:
                pt1 = points_2d[i]
                pt2 = points_2d[j]
                cv2.line(view_bg, pt1, pt2, self._colors['skeleton'], 2)

    # 绘制关节点
    for pt in points_2d:
        if pt is not None:
            cv2.circle(view_bg, pt, 3, self._colors['joints'], -1)

    # 绘制坐标轴
    if self._show_axes:
        # 坐标原点
        origin = (view_size // 2, view_size // 2)
        # X轴
        x_end = (origin[0] + int(50 * math.cos(rotation_rad)),
                 origin[1] - int(
                     50 * math.sin(elevation_rad) * math.sin(rotation_rad)))
        cv2.arrowedLine(view_bg, origin, x_end, self._colors['axes'][0], 2)

        # Y轴
        y_end = (
        origin[0] + int(50 * math.sin(rotation_rad) * math.sin(elevation_rad)),
        origin[1] - int(50 * math.cos(elevation_rad)))
        cv2.arrowedLine(view_bg, origin, y_end, self._colors['axes'][1], 2)

        # Z轴
        z_end = (origin[0] + int(50 * math.sin(rotation_rad)),
                 origin[1] + int(
                     50 * math.cos(rotation_rad) * math.sin(elevation_rad)))
        cv2.arrowedLine(view_bg, origin, z_end, self._colors['axes'][2], 2)

    # 将3D视图叠加到原始帧
    # 先绘制边框
    cv2.rectangle(frame, (view_x - 2, view_y - 2),
                  (view_x + view_size + 2, view_y + view_size + 2),
                  (255, 255, 255), 2)

    # 叠加视图
    frame[view_y:view_y + view_size, view_x:view_x + view_size] = view_bg

    # 添加标题
    cv2.putText(frame, "3D View", (view_x, view_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def _draw_trail_on_room(self, room_viz):
    """
    在房间视图上绘制轨迹

    Args:
        room_viz: 房间视图

    Returns:
        ndarray: 带轨迹的房间视图
    """
    if not self._trail_points or len(self._trail_points) < 2:
        return room_viz

    # 创建轨迹颜色渐变
    trail_colors = []
    for i in range(len(self._trail_points)):
        ratio = i / max(1, len(self._trail_points) - 1)
        # 从起始色到结束色的渐变
        start_color = np.array(self._colors['trail'][0])
        end_color = np.array(self._colors['trail'][1])
        color = tuple(map(int, start_color * (1 - ratio) + end_color * ratio))
        trail_colors.append(color)

    # 绘制轨迹线
    for i in range(1, len(self._trail_points)):
        pt1 = (
        int(self._trail_points[i - 1][0]), int(self._trail_points[i - 1][1]))
        pt2 = (int(self._trail_points[i][0]), int(self._trail_points[i][1]))
        cv2.line(room_viz, pt1, pt2, trail_colors[i], 2)

    # 绘制轨迹点
    for i, point in enumerate(self._trail_points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(room_viz, (x, y), 3, trail_colors[i], -1)

    return room_viz


def _draw_position_on_room(self, room_viz, position, action=None):
    """
    在房间视图上绘制当前位置

    Args:
        room_viz: 房间视图
        position: 3D位置 (x, y, z)
        action: 当前动作

    Returns:
        ndarray: 带位置标记的房间视图
    """
    x, y, z = position

    # 绘制位置标记
    cv2.circle(room_viz, (int(x), int(y)), 10, self._colors['position'], -1)

    # 根据深度绘制深度指示圆
    if z > 0:
        # 半透明圆，大小与深度成比例
        depth_radius = int(z * 5)  # 简单比例因子
        overlay = room_viz.copy()
        cv2.circle(overlay, (int(x), int(y)), depth_radius, (100, 100, 255), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, room_viz, 1 - alpha, 0, room_viz)

        # 添加深度标签
        cv2.putText(room_viz, f"Depth: {z:.1f}m", (int(x) + 15, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 如果有动作标签，显示在位置旁边
    if action:
        cv2.putText(room_viz, action, (int(x) + 15, int(y) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return room_viz


def _add_top_view(self, room_viz, keypoints_3d):
    """
    添加俯视图到房间视图

    Args:
        room_viz: 房间视图
        keypoints_3d: 3D关键点列表

    Returns:
        ndarray: 带俯视图的房间视图
    """
    # 设置俯视图区域
    view_size = 150
    margin = 20
    view_x = room_viz.shape[1] - view_size - margin
    view_y = margin

    # 创建俯视图背景
    view_bg = np.ones((view_size, view_size, 3), dtype=np.uint8) * 230

    # 绘制网格
    grid_spacing = view_size // 4
    for i in range(1, 4):
        # 水平线
        cv2.line(view_bg, (0, i * grid_spacing), (view_size, i * grid_spacing),
                 (200, 200, 200), 1)
        # 垂直线
        cv2.line(view_bg, (i * grid_spacing, 0), (i * grid_spacing, view_size),
                 (200, 200, 200), 1)

    # 投影关键点 (俯视图，x-z平面)
    points_2d = []
    for kp in keypoints_3d:
        if len(kp) >= 4 and kp[3] > 0.2:
            x, y, z = kp[0], kp[1], kp[2]

            # 缩放到视图大小
            scale = view_size / 300.0
            px = int(view_size / 2 + x * scale)
            pz = int(view_size / 2 + z * scale)

            points_2d.append((px, pz))
        else:
            points_2d.append(None)

    # 绘制骨架连接
    for conn in self._skeleton_connections:
        i, j = conn
        if i < len(points_2d) and j < len(points_2d):
            if points_2d[i] is not None and points_2d[j] is not None:
                cv2.line(view_bg, points_2d[i], points_2d[j], (100, 100, 200),
                         2)

    # 绘制关节点
    for pt in points_2d:
        if pt is not None:
            cv2.circle(view_bg, pt, 3, (200, 100, 100), -1)

    # 添加边框
    cv2.rectangle(view_bg, (0, 0), (view_size - 1, view_size - 1),
                  (100, 100, 100), 2)

    # 叠加到房间视图
    room_viz[view_y:view_y + view_size, view_x:view_x + view_size] = view_bg

    # 添加标题
    cv2.putText(room_viz, "Top View", (view_x, view_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return room_viz


def _adjust_keypoints_to_position(self, keypoints_3d, position):
    """
    调整3D关键点到指定位置

    Args:
        keypoints_3d: 原始3D关键点
        position: 目标位置

    Returns:
        list: 调整后的3D关键点
    """
    if not keypoints_3d:
        return None

    # 计算关键点中心
    valid_points = [kp for kp in keypoints_3d if len(kp) >= 4 and kp[3] > 0.2]
    if not valid_points:
        return keypoints_3d

    center_x = sum(p[0] for p in valid_points) / len(valid_points)
    center_y = sum(p[1] for p in valid_points) / len(valid_points)
    center_z = sum(p[2] for p in valid_points) / len(valid_points)

    # 计算偏移量
    offset_x = position[0] - center_x
    offset_y = position[1] - center_y
    offset_z = position[2] - center_z

    # 应用偏移
    adjusted = []
    for kp in keypoints_3d:
        if len(kp) >= 4:
            new_kp = [kp[0] + offset_x, kp[1] + offset_y, kp[2] + offset_z,
                      kp[3]]
            adjusted.append(new_kp)
        else:
            adjusted.append(kp)

    return adjusted


# 插件系统工厂方法
def create_plugin(plugin_id="3d_visualizer", config=None,
                  context=None) -> PluginInterface:
    """
    创建3D可视化器插件实例

    此函数是插件系统识别和加载插件的入口点

    Args:
        plugin_id: 插件唯一标识符
        config: 插件配置
        context: 插件上下文

    Returns:
        PluginInterface: 插件实例
    """
    try:
        # 创建插件实例
        plugin = ThreeDVisualizerPlugin(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建3D可视化器插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建3D可视化器插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None  # -*- coding: utf-8 -*-


"""
3D可视化器插件模块 - 提供3D人体姿态和空间可视化功能

此模块提供了一个3D可视化器插件，能够以三维方式展示人体姿态、
空间位置和轨迹，支持多角度观看和交互式显示。
"""

import logging
import time
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import math
import os

# 导入插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    VisualizerPluginInterface
)

# 导入日志配置
from utils.logger_config import get_logger

logger = get_logger("3DVisualizerPlugin")


class ThreeDVisualizerPlugin(VisualizerPluginInterface):
    """
    3D可视化器插件 - 提供三维可视化功能

    此插件能够以三维方式展示人体姿态、空间位置和轨迹，
    支持多角度观看和交互式显示。
    """

    def __init__(self, plugin_id="3d_visualizer", plugin_config=None):
        """
        初始化3D可视化器插件

        Args:
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "3D Visualizer"
        self._version = "1.0.0"
        self._description = "提供三维人体姿态和空间可视化功能的插件"
        self._config = plugin_config or {}

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 可视化器相关属性
        self._room_width = self._config.get('room_width', 500)  # 虚拟房间宽度(像素)
        self._room_height = self._config.get('room_height', 400)  # 虚拟房间高度(像素)
        self._room_depth = self._config.get('room_depth', 800)  # 虚拟房间深度(像素)

        # 视图设置
        self._view_angle = self._config.get('view_angle', 45)  # 视图角度(度)
        self._rotation_angle = self._config.get('rotation_angle', 30)  # 旋转角度(度)
        self._elevation_angle = self._config.get('elevation_angle', 30)  # 仰角(度)
        self._zoom_factor = self._config.get('zoom_factor', 1.0)  # 缩放因子

        # 轨迹设置
        self._trail_enabled = self._config.get('trail_enabled', True)  # 是否启用轨迹
        self._trail_length = self._config.get('trail_length', 50)  # 轨迹长度
        self._trail_points = []  # 轨迹点 [(x, y, z), ...]

        # 配色设置
        self._color_scheme = self._config.get('color_scheme', 'default')  # 配色方案
        self._colors = self._setup_colors()  # 设置颜色

        # 显示选项
        self._show_grid = self._config.get('show_grid', True)  # 是否显示网格
        self._show_axes = self._config.get('show_axes', True)  # 是否显示坐标轴
        self._show_skeleton = self._config.get('show_skeleton', True)  # 是否显示骨架
        self._show_floor = self._config.get('show_floor', True)  # 是否显示地面
        self._show_debug_info = self._config.get('show_debug_info',
                                                 True)  # 是否显示调试信息

        # 3D模型和姿态设置
        self._skeleton_connections = self._setup_skeleton()  # 建立骨架连接
        self._last_keypoints_3d = None  # 上次的3D关键点

        # 事件系统引用
        self._event_system = None

        # 缓存
        self._room_bg = None  # 房间背景图缓存
        self._last_frame_viz = None  # 上一帧可视化缓存

        # 动作标签设置
        self._action_label_pos = self._config.get('action_label_pos',
                                                  (20, 30))  # 动作标签位置
        self._action_label_size = self._config.get('action_label_size',
                                                   0.8)  # 动作标签大小

        logger.info(f"3D可视化器插件已创建: {plugin_id}")

    # ============= 实现PluginInterface基本方法 =============

    @property
    def id(self) -> str:
        """获取插件ID"""
        return self._id

    @property
    def name(self) -> str:
        """获取插件名称"""
        return self._name

    @property
    def version(self) -> str:
        """获取插件版本"""
        return self._version

    @property
    def description(self) -> str:
        """获取插件描述"""
        return self._description

    def get_dependencies(self) -> list:
        """获取插件依赖项列表"""
        return []  # 3D可视化器通常不依赖其他插件

    def is_initialized(self) -> bool:
        """检查插件是否已初始化"""
        return self._initialized

    def is_enabled(self) -> bool:
        """检查插件是否已启用"""
        return self._enabled

    def initialize(self, context: Dict[str, Any] = None) -> bool:
        """初始化插件

        Args:
            context: 插件初始化上下文，可以包含共享资源

        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.is_initialized():
                logger.warning(f"插件 {self._id} 已初始化，跳过")
                return True

            # 创建房间背景
            self._create_room_bg()

            # 获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']
                logger.info("已设置事件系统")

                # 发布初始化事件
                if hasattr(self._event_system, 'publish'):
                    self._event_system.publish(
                        "plugin_initialized",
                        {
                            'plugin_id': self._id,
                            'plugin_type': 'visualizer'
                        }
                    )

            self._initialized = True
            logger.info(f"插件 {self._id} 初始化成功")
            return True

        except Exception as e:
            logger.error(f"插件 {self._id} 初始化失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def enable(self) -> bool:
        """启用插件

        Returns:
            bool: 启用是否成功
        """
        if not self.is_initialized():
            logger.error(f"插件 {self._id} 未初始化，无法启用")
            return False

        try:
            # 执行启用逻辑
            self._enabled = True
            logger.info(f"插件 {self._id} 已启用")

            # 发布启用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_enabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'visualizer'
                    }
                )

            return True
        except Exception as e:
            logger.error(f"启用插件 {self._id} 时出错: {e}")
            return False

    def disable(self) -> bool:
        """禁用插件

        Returns:
            bool: 禁用是否成功
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 已经是禁用状态")
            return True

        try:
            # 执行禁用逻辑
            self._enabled = False
            logger.info(f"插件 {self._id} 已禁用")

            # 发布禁用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_disabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'visualizer'
                    }
                )

            return True
        except Exception as e:
            logger.error(f"禁用插件 {self._id} 时出错: {e}")
            return False

    def configure(self, config: Dict[str, Any]) -> bool:
        """配置插件

        Args:
            config: 插件配置参数

        Returns:
            bool: 配置是否成功
        """
        try:
            # 更新配置
            self._config.update(config)

            # 更新可视化器参数
            if 'room_width' in config:
                self._room_width = config['room_width']
                # 重新创建房间背景
                self._create_room_bg()

            if 'room_height' in config:
                self._room_height = config['room_height']
                # 重新创建房间背景
                self._create_room_bg()

            if 'room_depth' in config:
                self._room_depth = config['room_depth']

            if 'view_angle' in config:
                self._view_angle = config['view_angle']

            if 'rotation_angle' in config:
                self._rotation_angle = config['rotation_angle']

            if 'elevation_angle' in config:
                self._elevation_angle = config['elevation_angle']

            if 'zoom_factor' in config:
                self._zoom_factor = config['zoom_factor']

            if 'trail_enabled' in config:
                self._trail_enabled = config['trail_enabled']

            if 'trail_length' in config:
                self._trail_length = config['trail_length']
                # 如果轨迹长度已经超过新的长度，进行截断
                if len(self._trail_points) > self._trail_length:
                    self._trail_points = self._trail_points[
                                         -self._trail_length:]

            if 'color_scheme' in config:
                self._color_scheme = config['color_scheme']
                self._colors = self._setup_colors()

            if 'show_grid' in config:
                self._show_grid = config['show_grid']

            if 'show_axes' in config:
                self._show_axes = config['show_axes']

            if 'show_skeleton' in config:
                self._show_skeleton = config['show_skeleton']

            if 'show_floor' in config:
                self._show_floor = config['show_floor']

            if 'show_debug_info' in config:
                self._show_debug_info = config['show_debug_info']

            # 记录日志
            logger.info(f"插件 {self._id} 配置已更新")
            return True
        except Exception as e:
            logger.error(f"配置插件 {self._id} 时出错: {e}")
            return False

    def cleanup(self) -> bool:
        """清理插件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            # 执行清理逻辑
            self._enabled = False
            self._initialized = False

            # 清理资源
            self._trail_points.clear()
            self._room_bg = None
            self._last_frame_viz = None

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 实现VisualizerPluginInterface特定方法 =============

    def visualize_frame(self, frame, person=None, action=None, detector=None):
        """
        可视化相机帧

        Args:
            frame: 相机帧图像
            person: 检测到的人体数据
            action: 识别的动作
            detector: 检测器实例

        Returns:
            ndarray: 带有可视化效果的帧
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法可视化帧")
            return frame

        try:
            # 创建帧副本
            viz_frame = frame.copy()

            # 如果有人体数据
            if person and 'keypoints' in person:
                keypoints = person.get('keypoints', [])

                # 绘制2D骨架
                if self._show_skeleton and keypoints:
                    viz_frame = self._draw_2d_skeleton(viz_frame, keypoints,
                                                       action)

                # 从2D关键点生成3D关键点
                keypoints_3d = self._estimate_3d_keypoints(keypoints)
                self._last_keypoints_3d = keypoints_3d

                # 在图像角落绘制3D视图
                if self._last_keypoints_3d is not None:
                    # 在右上角创建3D视图
                    viz_frame = self._add_3d_view(viz_frame, keypoints_3d,
                                                  action)

            # 添加动作标签
            if action:
                cv2.putText(viz_frame, f"Action: {action}",
                            self._action_label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            self._action_label_size, (0, 255, 0), 2)

            # 缓存当前可视化
            self._last_frame_viz = viz_frame

            return viz_frame

        except Exception as e:
            logger.error(f"可视化帧时出错: {e}")
            # 如果有缓存的上一帧可视化，则返回它
            if self._last_frame_viz is not None:
                return self._last_frame_viz
            return frame

    def visualize_room(self, position=None, depth=None, action=None):
        """
        可视化房间地图

        Args:
            position: 房间中的(x,y)位置
            depth: 深度/距离估计
            action: 识别的动作

        Returns:
            ndarray: 房间可视化图像
        """
        if not self.is_enabled():
            logger.warning(f"插件 {self._id} 未启用，无法可视化房间")
            return None

        try:
            # 确保房间背景已创建
            if self._room_bg is None:
                self._create_room_bg()

            # 创建房间视图副本
            room_viz = self._room_bg.copy()

            # 如果有位置信息
            if position is not None:
                # 创建3D位置
                pos_3d = [position[0], position[1], depth or 0]

                # 添加到轨迹
                if self._trail_enabled:
                    self._add_trail_point(*pos_3d)

                    # 绘制轨迹
                    if self._trail_points:
                        room_viz = self._draw_trail_on_room(room_viz)

                # 绘制当前位置
                self._draw_position_on_room(room_viz, pos_3d, action)

            # 添加动作标签
            if action:
                cv2.putText(room_viz, f"Action: {action}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 添加3D视图
            if self._last_keypoints_3d is not None and position is not None:
                # 更新3D关键点位置
                adjusted_keypoints = self._adjust_keypoints_to_position(
                    self._last_keypoints_3d, pos_3d)

                # 绘制3D视图
                room_viz = self._add_top_view(room_viz, adjusted_keypoints)

            return room_viz

        except Exception as e:
            logger.error(f"可视化房间时出错: {e}")
            if self._room_bg is not None:
                return self._room_bg
            return None

    def add_trail_point(self, x, y
