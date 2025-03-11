import cv2
import numpy as np
import math
from utils.data_structures import CircularBuffer


class Visualizer:
    """负责可视化，更新以支持SimplifiedPersonDetector"""

    def __init__(self, room_width=800, room_height=600):
        # 创建房间平面图
        self.room_template = self._create_room_template(room_width, room_height)
        self.room_width = room_width
        self.room_height = room_height

        # 轨迹点
        self.trail_points = CircularBuffer(20)

    def _create_room_template(self, width, height):
        """创建房间平面图模板"""
        template = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 绘制网格线
        for x in range(0, width, 100):
            cv2.line(template, (x, 0), (x, height), (220, 220, 220), 1)
        for y in range(0, height, 100):
            cv2.line(template, (0, y), (width, y), (220, 220, 220), 1)

        # 绘制中心坐标轴
        cv2.line(template, (width // 2, 0), (width // 2, height), (0, 0, 255),
                 2)
        cv2.line(template, (0, height // 2), (width, height // 2), (0, 0, 255),
                 2)

        # 添加深度标签
        cv2.putText(template, "NEAR", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(template, "FAR", (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 标记房间顶部区域（近）和底部区域（远）
        cv2.line(template, (0, 50), (width, 50), (100, 100, 255), 2)
        cv2.line(template, (0, height - 50), (width, height - 50),
                 (255, 100, 100), 2)

        return template

    def add_trail_point(self, x, y):
        """添加轨迹点"""
        if len(self.trail_points) == 0 or math.dist(self.trail_points[-1],
                                                    (x, y)) > 5:
            self.trail_points.append((x, y))

    def visualize_room(self, position=None, depth=None, action=None):
        """可视化房间平面图"""
        # 创建房间平面图副本
        try:
            room_viz = self.room_template.copy()
            # 绘制历史轨迹
            if hasattr(self, 'trail_points') and len(self.trail_points) > 1:
                for i in range(1, len(self.trail_points)):
                    try:
                        cv2.line(room_viz, self.trail_points[i - 1],
                                 self.trail_points[i],
                                 (200, 200, 200), 1)  # 使用浅灰色
                    except Exception:
                        pass  # 忽略单个线条的绘制错误

            # 绘制当前位置
            if position:
                x, y = position
                radius = max(15, int(30 - (depth or 0) / 2))

                # 在房间平面图上绘制人体位置
                cv2.circle(room_viz, (x, y), radius, (0, 0, 255), -1)

                # 显示深度信息（如果有）
                if depth is not None:
                    depth_text = f"Depth: {depth:.1f}"
                    cv2.putText(room_viz, depth_text, (x - 50, y - radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # 显示动作注释（如果有）
                if action and action not in ["Moving", "Static",
                                             "Collecting data..."]:
                    cv2.putText(room_viz, action, (x + radius + 10, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            return room_viz
        except Exception as e:
            print(f"可视化房间错误: {e}")
            return np.ones((self.room_height, self.room_width, 3),
                           dtype=np.uint8) * 255

    def visualize_frame(self, frame, person=None, action=None, detector=None):
        """可视化摄像头帧 - 兼容SimplifiedPersonDetector"""
        frame_viz = frame.copy()

        if person:
            # 绘制边界框
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 显示高度信息
            height_text = f"Height: {person['height']:.1f}"
            cv2.putText(frame_viz, height_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 绘制骨骼（如果有检测器和关键点）
            if detector and 'keypoints' in person:
                # 检查SimplifiedPersonDetector
                if hasattr(detector, 'draw_skeleton'):
                    # 传统接口和简化检测器
                    detector.draw_skeleton(frame_viz, person['keypoints'])
                elif detector.__class__.__name__ == 'SimplifiedPersonDetector':
                    # 如果是SimplifiedPersonDetector但没有直接暴露draw_skeleton方法
                    # 则调用特殊处理
                    self._draw_simplified_skeleton(frame_viz,
                                                   person['keypoints'],
                                                   detector)
                else:
                    # 基本骨架绘制逻辑
                    self._draw_basic_skeleton(frame_viz, person['keypoints'])

        # 显示动作
        if action:
            action_text = f"Action: {action}"
            cv2.putText(frame_viz, action_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame_viz

    def _draw_simplified_skeleton(self, frame, keypoints, detector):
        """为SimplifiedPersonDetector绘制骨骼"""
        # 使用SimplifiedPersonDetector的内部方法
        if hasattr(detector, '_draw_skeleton'):
            detector._draw_skeleton(frame, keypoints)
        else:
            self._draw_basic_skeleton(frame, keypoints)

    def _draw_basic_skeleton(self, frame, keypoints):
        """基本的骨骼绘制逻辑，作为后备"""
        # 基本连接定义
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 面部和颈部
            (5, 6), (5, 11), (6, 12), (11, 12),  # 躯干
            (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
        ]

        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:  # 使用基本置信度阈值
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # 绘制连接
        for connection in connections:
            p1, p2 = connection
            if p1 < len(keypoints) and p2 < len(keypoints):
                if keypoints[p1][2] > 0.5 and keypoints[p2][2] > 0.5:
                    cv2.line(frame,
                             (int(keypoints[p1][0]), int(keypoints[p1][1])),
                             (int(keypoints[p2][0]), int(keypoints[p2][1])),
                             (0, 255, 255), 2)
