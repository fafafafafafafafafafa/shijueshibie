"""
数学运算工具模块
提供共用的数学计算和平滑处理功能，供各模块使用
"""
import math
import numpy as np


class MathUtils:
    """提供各种数学计算和平滑处理的工具类"""

    @staticmethod
    def distance(point1, point2):
        """计算两点之间的欧几里得距离

        Args:
            point1: 第一个点坐标 (x1, y1)
            point2: 第二个点坐标 (x2, y2)

        Returns:
            float: 两点之间的距离
        """
        x1, y1 = point1
        x2, y2 = point2
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def distance_squared(point1, point2):
        """计算两点之间距离的平方（避免开平方运算）

        Args:
            point1: 第一个点坐标 (x1, y1)
            point2: 第二个点坐标 (x2, y2)

        Returns:
            float: 两点之间距离的平方
        """
        x1, y1 = point1
        x2, y2 = point2
        dx = x2 - x1
        dy = y2 - y1
        return dx * dx + dy * dy

    @staticmethod
    def clamp(value, min_value, max_value):
        """将值限制在指定范围内

        Args:
            value: 要限制的值
            min_value: 最小允许值
            max_value: 最大允许值

        Returns:
            与输入相同类型: 限制在范围内的值
        """
        return max(min_value, min(value, max_value))

    @staticmethod
    def linear_interpolate(value1, value2, factor):
        """线性插值

        Args:
            value1: 第一个值
            value2: 第二个值
            factor: 插值因子 (0.0 - 1.0)

        Returns:
            与输入相同类型: 插值后的值
        """
        return value1 * (1 - factor) + value2 * factor

    @staticmethod
    def sigmoid(x, k=1.0):
        """计算Sigmoid函数值

        Args:
            x: 输入值
            k: 控制曲线陡峭度的系数

        Returns:
            float: Sigmoid函数值 (0.0 - 1.0)
        """
        return 1.0 / (1.0 + math.exp(-k * x))

    @staticmethod
    def sigmoid_blend(progress, k=6.0):
        """使用Sigmoid函数创建非线性混合因子

        Args:
            progress: 进度 (0.0 - 1.0)
            k: 控制曲线陡峭度的系数

        Returns:
            float: 非线性混合因子 (0.0 - 1.0)
        """
        return 1.0 / (1.0 + math.exp(-k * (progress - 0.5)))


class SmoothingAlgorithms:
    """提供各种平滑算法的类"""

    @staticmethod
    def simple_moving_average(points, window_size=None):
        """简单移动平均

        Args:
            points: 点坐标序列 [(x1,y1), (x2,y2), ...]
            window_size: 窗口大小，None则使用全部点

        Returns:
            tuple: 平滑后的坐标 (x, y)
        """
        if not points:
            return None

        if window_size is None or window_size > len(points):
            window_size = len(points)

        # 只取最近的点
        recent_points = points[-window_size:]

        # 计算平均值
        avg_x = sum(p[0] for p in recent_points) / len(recent_points)
        avg_y = sum(p[1] for p in recent_points) / len(recent_points)

        return int(avg_x), int(avg_y)

    @staticmethod
    def weighted_moving_average(points, weights=None):
        """加权移动平均

        Args:
            points: 点坐标序列 [(x1,y1), (x2,y2), ...]
            weights: 权重序列，None则使用线性递增权重

        Returns:
            tuple: 平滑后的坐标 (x, y)
        """
        if not points:
            return None

        # 如果未提供权重，使用线性递增权重
        if weights is None:
            weights = [i + 1 for i in range(len(points))]

        # 确保权重和点的数量一致
        if len(weights) != len(points):
            weights = weights[:len(points)] if len(weights) > len(
                points) else weights + [weights[-1]] * (
                        len(points) - len(weights))

        # 计算加权平均
        total_x = sum(p[0] * w for p, w in zip(points, weights))
        total_y = sum(p[1] * w for p, w in zip(points, weights))
        total_weight = sum(weights)

        if total_weight > 0:
            avg_x = total_x / total_weight
            avg_y = total_y / total_weight
            return int(avg_x), int(avg_y)

        return points[-1]  # 返回最后一个点

    @staticmethod
    def exponential_moving_average(points, alpha=0.3, initial=None):
        """指数移动平均

        Args:
            points: 点坐标序列 [(x1,y1), (x2,y2), ...]
            alpha: 平滑因子 (0-1)，较大的值响应更快
            initial: 初始值，None则使用第一个点

        Returns:
            tuple: 平滑后的坐标 (x, y)
        """
        if not points:
            return None

        if initial is None:
            ema_x, ema_y = points[0]
            start_idx = 1
        else:
            ema_x, ema_y = initial
            start_idx = 0

        # 计算EMA
        for i in range(start_idx, len(points)):
            ema_x = alpha * points[i][0] + (1 - alpha) * ema_x
            ema_y = alpha * points[i][1] + (1 - alpha) * ema_y

        return int(ema_x), int(ema_y)

    @staticmethod
    def gaussian_weighted_average(points, center_idx=None, sigma=None):
        """高斯加权平均

        Args:
            points: 点坐标序列 [(x1,y1), (x2,y2), ...]
            center_idx: 高斯中心点索引，None则使用中心位置
            sigma: 高斯分布的标准差，None则使用序列长度/3

        Returns:
            tuple: 平滑后的坐标 (x, y)
        """
        if not points or len(points) < 2:
            return None if not points else points[-1]

        # 默认高斯中心在60%处
        if center_idx is None:
            center_idx = int(len(points) * 0.6)

        # 默认sigma
        if sigma is None:
            sigma = len(points) / 3.0

        # 计算高斯权重
        total_x, total_y, total_weight = 0, 0, 0

        for i, (px, py) in enumerate(points):
            # 计算高斯权重
            weight = math.exp(-0.5 * ((i - center_idx) / sigma) ** 2)
            total_x += px * weight
            total_y += py * weight
            total_weight += weight

        if total_weight > 0:
            avg_x = total_x / total_weight
            avg_y = total_y / total_weight
            return int(avg_x), int(avg_y)

        return points[-1]  # 返回最后一个点

    @staticmethod
    def blended_average(current, historical, blend_factor=0.5):
        """混合当前值与历史平均

        Args:
            current: 当前点坐标 (x, y)
            historical: 历史平均点坐标 (x, y)
            blend_factor: 混合因子 (0-1)，表示当前值的权重

        Returns:
            tuple: 混合后的坐标 (x, y)
        """
        if not historical:
            return current

        x, y = current
        hist_x, hist_y = historical

        smooth_x = int(x * blend_factor + hist_x * (1 - blend_factor))
        smooth_y = int(y * blend_factor + hist_y * (1 - blend_factor))

        return smooth_x, smooth_y

    @staticmethod
    def kalman_filter_1d(measurements, process_variance=1e-4,
                         measurement_variance=1e-2):
        """一维卡尔曼滤波

        Args:
            measurements: 测量值序列
            process_variance: 过程方差
            measurement_variance: 测量方差

        Returns:
            list: 滤波后的值序列
        """
        n = len(measurements)
        if n == 0:
            return []

        # 卡尔曼滤波状态
        x_hat = measurements[0]  # 初始状态估计
        p = 1.0  # 初始协方差估计

        # 结果序列
        filtered = [x_hat]

        # 滤波过程
        for z in measurements[1:]:
            # 预测
            p = p + process_variance

            # 更新
            k = p / (p + measurement_variance)  # 卡尔曼增益
            x_hat = x_hat + k * (z - x_hat)
            p = (1 - k) * p

            filtered.append(x_hat)

        return filtered
