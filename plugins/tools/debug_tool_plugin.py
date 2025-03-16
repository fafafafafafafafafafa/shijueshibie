import psutil

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
self._memory_usage.append(memory_mb)
except ImportError:
# 如果没有psutil，忽略内存监控
pass

# 记录系统事件
event = {
    'timestamp': time.time(),
    'fps': self._fps_history[-1] if self._fps_history else 0,
    'memory_mb': self._memory_usage[-1] if self._memory_usage else 0
}

self._system_summary['system_events'].append(event)

except Exception as e:
logger.error(f"记录周期性数据时出错: {e}")


def _detect_anomalies(self, metrics):
    """
    检测性能异常

    Args:
        metrics: 性能指标数据
    """
    try:
        # 检测FPS异常
        if 'fps' in metrics and metrics['fps'] < self._fps_threshold:
            warning = {
                'timestamp': time.time(),
                'type': 'low_fps',
                'value': metrics['fps'],
                'threshold': self._fps_threshold,
                'message': f"FPS低于阈值: {metrics['fps']:.1f} < {self._fps_threshold:.1f}"
            }

            self._system_summary['warnings'].append(warning)
            logger.warning(warning['message'])

            # 发布警告事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "performance_warning",
                    {
                        'plugin_id': self._id,
                        'warning_type': 'low_fps',
                        'fps': metrics['fps'],
                        'threshold': self._fps_threshold
                    }
                )

        # 检测帧处理延迟异常
        if 'frame_time' in metrics and metrics[
            'frame_time'] * 1000 > self._latency_threshold:
            warning = {
                'timestamp': time.time(),
                'type': 'high_latency',
                'value': metrics['frame_time'] * 1000,
                'threshold': self._latency_threshold,
                'message': f"帧处理延迟过高: {metrics['frame_time'] * 1000:.1f}ms > {self._latency_threshold:.1f}ms"
            }

            self._system_summary['warnings'].append(warning)
            logger.warning(warning['message'])

            # 发布警告事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "performance_warning",
                    {
                        'plugin_id': self._id,
                        'warning_type': 'high_latency',
                        'latency_ms': metrics['frame_time'] * 1000,
                        'threshold': self._latency_threshold
                    }
                )

    except Exception as e:
        logger.error(f"检测异常时出错: {e}")


def _take_snapshot(self):
    """生成系统快照"""
    try:
        # 收集当前状态
        snapshot = {
            'timestamp': time.time(),
            'fps': self._fps_history[-1] if self._fps_history else 0,
            'memory_mb': self._memory_usage[-1] if self._memory_usage else 0,
            'frame_time_ms': self._frame_times[
                                 -1] * 1000 if self._frame_times else 0,
            'component_states': self._component_states.copy(),
            'frames_processed': self._system_summary['frames_processed'],
            'detections': self._system_summary['detections'],
            'actions_recognized': self._system_summary['actions_recognized']
        }

        # 生成快照文件名
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_file = os.path.join(self._report_dir,
                                     f"snapshot_{timestamp}.json")

        # 保存快照
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"系统快照已保存: {snapshot_file}")

    except Exception as e:
        logger.error(f"生成系统快照时出错: {e}")


def _collect_report_data(self, final=False):
    """
    收集报告数据

    Args:
        final: 是否为最终报告

    Returns:
        Dict: 报告数据
    """
    try:
        # 计算运行时间
        run_time = time.time() - self._system_summary['start_time']

        report = {
            'timestamp': time.time(),
            'plugin_version': self._version,
            'run_time_seconds': run_time,
            'frames_processed': self._system_summary['frames_processed'],
            'average_fps': sum(self._fps_history) / len(
                self._fps_history) if self._fps_history else 0,
            'detections': self._system_summary['detections'],
            'actions_recognized': self._system_summary['actions_recognized'],
            'warnings_count': len(self._system_summary['warnings']),
            'errors_count': len(self._system_summary['errors'])
        }

        # 性能分析
        report['performance_analysis'] = self.analyze_performance()

        # 对于最终报告，包含完整的警告和错误列表
        if final:
            report['warnings'] = self._system_summary['warnings']
            report['errors'] = self._system_summary['errors']
            report['component_states'] = self._component_states

            # 仅包含最近100个系统事件
            report['recent_events'] = self._system_summary['system_events'][
                                      -100:] if self._system_summary[
                'system_events'] else []

            # 添加建议
            report['recommendations'] = self._generate_recommendations()

        return report

    except Exception as e:
        logger.error(f"收集报告数据时出错: {e}")
        return {'error': str(e)}


def _generate_recommendations(self):
    """
    生成系统优化建议

    Returns:
        List: 建议列表
    """
    recommendations = []

    try:
        # 基于FPS的建议
        if self._fps_history:
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            min_fps = min(self._fps_history)

            if min_fps < 10:
                recommendations.append({
                    'type': 'critical',
                    'area': 'performance',
                    'message': f"FPS非常低 ({min_fps:.1f})，建议降低处理分辨率或简化处理管道"
                })
            elif avg_fps < 20:
                recommendations.append({
                    'type': 'warning',
                    'area': 'performance',
                    'message': f"平均FPS较低 ({avg_fps:.1f})，可能影响用户体验"
                })

        # 检测组件瓶颈
        if self._component_analysis and 'time_distribution' in self.analyze_performance():
            time_dist = self.analyze_performance()['time_distribution']
            for component, data in time_dist.items():
                if data['percentage'] > 60:
                    recommendations.append({
                        'type': 'warning',
                        'area': 'bottleneck',
                        'message': f"组件 '{component}' 占用了处理时间的 {data['percentage']:.1f}%，是主要瓶颈"
                    })

        # 内存使用建议
        if self._memory_usage:
            max_memory = max(self._memory_usage)
            if max_memory > 1000:  # 超过1GB
                recommendations.append({
                    'type': 'warning',
                    'area': 'memory',
                    'message': f"内存使用峰值 ({max_memory:.1f}MB) 较高，建议优化内存使用或增加系统内存"
                })

        # 基于错误数的建议
        if len(self._system_summary['errors']) > 0:
            recommendations.append({
                'type': 'critical',
                'area': 'stability',
                'message': f"系统运行期间出现 {len(self._system_summary['errors'])} 个错误，建议检查日志并修复问题"
            })

    except Exception as e:
        logger.error(f"生成建议时出错: {e}")
        recommendations.append({
            'type': 'error',
            'area': 'system',
            'message': f"生成建议过程中出错: {e}"
        })

    return recommendations


def _overlay_metrics_on_frame(self, frame, metrics):
    """
    在帧上叠加性能指标

    Args:
        frame: 输入帧
        metrics: 性能指标

    Returns:
        ndarray: 带指标的帧
    """
    try:
        # 创建帧副本
        viz_frame = frame.copy()

        # 创建半透明背景
        overlay = viz_frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, viz_frame, 1 - alpha, 0, viz_frame)

        # 添加标题
        cv2.putText(viz_frame, "Performance Metrics", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 绘制指标
        y_pos = 60

        # FPS
        if 'fps' in metrics:
            fps_color = (0, 255, 0) if metrics[
                                           'fps'] >= self._fps_threshold else (
            0, 165, 255)
            cv2.putText(viz_frame, f"FPS: {metrics['fps']:.1f}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
            y_pos += 25

        # 帧时间
        if 'frame_time' in metrics:
            frame_ms = metrics['frame_time'] * 1000
            latency_color = (
            0, 255, 0) if frame_ms < self._latency_threshold else (0, 165, 255)
            cv2.putText(viz_frame, f"Frame Time: {frame_ms:.1f} ms",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, latency_color, 1)
            y_pos += 25

        # 检测时间
        if 'detection_time' in metrics:
            det_ms = metrics['detection_time'] * 1000
            cv2.putText(viz_frame, f"Detection: {det_ms:.1f} ms", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25

        # 动作识别时间
        if 'recognition_time' in metrics:
            recog_ms = metrics['recognition_time'] * 1000
            cv2.putText(viz_frame, f"Recognition: {recog_ms:.1f} ms",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25

        # 内存使用
        if 'memory' in metrics:
            mem_mb = metrics['memory']
            cv2.putText(viz_frame, f"Memory: {mem_mb:.1f} MB", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return viz_frame

    except Exception as e:
        logger.error(f"在帧上叠加指标时出错: {e}")
        return frame

    # ============= 事件处理方法 =============


def _on_fps_updated(self, data):
    """处理FPS更新事件"""
    if 'fps' in data:
        fps = data['fps']
        # 添加到历史记录
        self._fps_history.append(fps)


def _on_person_detected(self, data):
    """处理人体检测事件"""
    self._system_summary['detections'] += 1

    if 'detection_time' in data:
        # 记录检测时间
        self._detection_times.append(data['detection_time'])


def _on_action_recognized(self, data):
    """处理动作识别事件"""
    self._system_summary['actions_recognized'] += 1

    if 'recognition_time' in data:
        # 记录识别时间
        self._recognition_times.append(data['recognition_time'])


def _on_position_mapped(self, data):
    """处理位置映射事件"""
    if 'mapping_time' in data:
        # 记录映射时间
        self._mapping_times.append(data['mapping_time'])


def _on_system_state_changed(self, data):
    """处理系统状态变更事件"""
    if 'state' in data:
        state = data['state']
        component_id = data.get('component_id', 'system')

        # 更新组件状态
        if component_id not in self._component_states:
            self._component_states[component_id] = {}

        self._component_states[component_id]['state'] = state
        self._component_states[component_id]['timestamp'] = time.time()


def _on_error_occurred(self, data):
    """处理错误事件"""
    if 'error' in data:
        error = {
            'timestamp': time.time(),
            'message': data['error'],
            'component_id': data.get('component_id', 'unknown')
        }

        self._system_summary['errors'].append(error)

        # 更新组件错误计数
        component_id = data.get('component_id', 'unknown')
        if component_id not in self._component_states:
            self._component_states[component_id] = {'error_count': 0}

        self._component_states[component_id]['error_count'] = \
        self._component_states[component_id].get('error_count', 0) + 1


# 插件系统工厂方法
def create_plugin(plugin_id="debug_tool", config=None,
                  context=None) -> PluginInterface:
    """
    创建调试工具插件实例

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
        plugin = DebugToolPlugin(
            plugin_id=plugin_id,
            plugin_config=config
        )

        # 如果有上下文，初始化插件
        if context:
            plugin.initialize(context)

        logger.info(f"创建调试工具插件成功: {plugin_id}")
        return plugin
    except Exception as e:
        logger.error(f"创建调试工具插件失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None  # -*- coding: utf-8 -*-


"""
调试工具插件模块 - 提供系统调试和分析功能

此模块提供了一个调试工具插件，用于收集和分析系统性能指标、
检测潜在问题和生成诊断报告。
"""

import logging
import time
import os
import json
import datetime
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import deque

# 导入插件接口
from plugins.core.plugin_interface import (
    PluginInterface,
    ToolPluginInterface
)

# 导入日志配置
from utils.logger_config import get_logger

logger = get_logger("DebugToolPlugin")


class DebugToolPlugin(ToolPluginInterface):
    """
    调试工具插件 - 提供系统调试和分析功能

    此插件收集和分析系统性能指标、检测潜在问题并
    生成诊断报告。
    """

    def __init__(self, plugin_id="debug_tool", plugin_config=None):
        """
        初始化调试工具插件

        Args:
            plugin_id: 插件唯一标识符
            plugin_config: 插件配置参数
        """
        # 插件元数据
        self._id = plugin_id
        self._name = "Debug Tool"
        self._version = "1.0.0"
        self._description = "提供系统调试和分析功能的插件"
        self._config = plugin_config or {}

        # 插件状态
        self._initialized = False
        self._enabled = False

        # 调试工具相关属性
        self._metrics_enabled = self._config.get('metrics_enabled',
                                                 True)  # 是否收集性能指标
        self._log_level = self._config.get('log_level', 'INFO')  # 日志级别
        self._record_interval = self._config.get('record_interval',
                                                 1.0)  # 记录间隔(秒)
        self._report_dir = self._config.get('report_dir',
                                            'reports/debug')  # 报告保存目录

        # 性能指标历史
        self._frame_times = deque(maxlen=100)  # 帧处理时间
        self._fps_history = deque(maxlen=100)  # FPS历史
        self._memory_usage = deque(maxlen=100)  # 内存使用
        self._detection_times = deque(maxlen=100)  # 检测时间
        self._mapping_times = deque(maxlen=100)  # 映射时间
        self._recognition_times = deque(maxlen=100)  # 识别时间
        self._component_states = {}  # 组件状态

        # 异常检测设置
        self._anomaly_detection = self._config.get('anomaly_detection',
                                                   True)  # 是否启用异常检测
        self._fps_threshold = self._config.get('fps_threshold', 15.0)  # FPS警告阈值
        self._latency_threshold = self._config.get('latency_threshold',
                                                   100.0)  # 延迟警告阈值(毫秒)

        # 系统摘要
        self._system_summary = {
            'start_time': time.time(),
            'frames_processed': 0,
            'detections': 0,
            'actions_recognized': 0,
            'system_events': [],
            'warnings': [],
            'errors': []
        }

        # 可视化设置
        self._visualization_enabled = self._config.get('visualization_enabled',
                                                       True)  # 是否启用可视化
        self._overlay_metrics = self._config.get('overlay_metrics',
                                                 True)  # 是否在画面上叠加指标
        self._live_plot = self._config.get('live_plot', False)  # 是否启用实时图表

        # 事件系统引用
        self._event_system = None

        # 诊断工具
        self._bottleneck_analysis = self._config.get('bottleneck_analysis',
                                                     True)  # 瓶颈分析
        self._component_analysis = self._config.get('component_analysis',
                                                    True)  # 组件分析

        # 快照功能
        self._snapshot_enabled = self._config.get('snapshot_enabled',
                                                  True)  # 是否启用快照
        self._snapshot_interval = self._config.get('snapshot_interval',
                                                   60.0)  # 快照间隔(秒)
        self._last_snapshot_time = 0  # 上次快照时间

        # 资源监控
        self._resource_monitoring = self._config.get('resource_monitoring',
                                                     True)  # 是否监控资源

        # 定时器
        self._last_record_time = 0  # 上次记录时间

        logger.info(f"调试工具插件已创建: {plugin_id}")

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
        return []  # 调试工具通常不依赖其他插件

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

            # 设置日志级别
            logger.setLevel(getattr(logging, self._log_level))

            # 确保报告目录存在
            os.makedirs(self._report_dir, exist_ok=True)

            # 获取事件系统
            if context and 'event_system' in context:
                self._event_system = context['event_system']
                logger.info("已设置事件系统")

                # 设置事件订阅
                if hasattr(self._event_system, 'subscribe'):
                    self._setup_event_subscriptions()

                # 发布初始化事件
                if hasattr(self._event_system, 'publish'):
                    self._event_system.publish(
                        "plugin_initialized",
                        {
                            'plugin_id': self._id,
                            'plugin_type': 'tool',
                            'tool_type': 'debug'
                        }
                    )

            # 记录系统配置
            self._record_system_config(context)

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
            self._last_record_time = time.time()
            self._system_summary['start_time'] = time.time()

            logger.info(f"插件 {self._id} 已启用")

            # 发布启用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_enabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'tool',
                        'tool_type': 'debug'
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

            # 生成最终报告
            self.generate_report()

            logger.info(f"插件 {self._id} 已禁用")

            # 发布禁用事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "plugin_disabled",
                    {
                        'plugin_id': self._id,
                        'plugin_type': 'tool',
                        'tool_type': 'debug'
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

            # 更新调试工具参数
            if 'metrics_enabled' in config:
                self._metrics_enabled = config['metrics_enabled']

            if 'log_level' in config:
                self._log_level = config['log_level']
                logger.setLevel(getattr(logging, self._log_level))

            if 'record_interval' in config:
                self._record_interval = config['record_interval']

            if 'report_dir' in config:
                self._report_dir = config['report_dir']
                os.makedirs(self._report_dir, exist_ok=True)

            if 'anomaly_detection' in config:
                self._anomaly_detection = config['anomaly_detection']

            if 'fps_threshold' in config:
                self._fps_threshold = config['fps_threshold']

            if 'latency_threshold' in config:
                self._latency_threshold = config['latency_threshold']

            if 'visualization_enabled' in config:
                self._visualization_enabled = config['visualization_enabled']

            if 'overlay_metrics' in config:
                self._overlay_metrics = config['overlay_metrics']

            if 'live_plot' in config:
                self._live_plot = config['live_plot']

            if 'bottleneck_analysis' in config:
                self._bottleneck_analysis = config['bottleneck_analysis']

            if 'component_analysis' in config:
                self._component_analysis = config['component_analysis']

            if 'snapshot_enabled' in config:
                self._snapshot_enabled = config['snapshot_enabled']

            if 'snapshot_interval' in config:
                self._snapshot_interval = config['snapshot_interval']

            if 'resource_monitoring' in config:
                self._resource_monitoring = config['resource_monitoring']

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

            # 生成最终报告
            self.generate_report(final=True)

            # 清理资源
            self._frame_times.clear()
            self._fps_history.clear()
            self._memory_usage.clear()
            self._detection_times.clear()
            self._mapping_times.clear()
            self._recognition_times.clear()

            logger.info(f"插件 {self._id} 已清理")
            return True
        except Exception as e:
            logger.error(f"清理插件 {self._id} 时出错: {e}")
            return False

    # ============= 实现ToolPluginInterface特定方法 =============

    def process_frame(self, frame, metrics):
        """
        处理帧和相关指标

        Args:
            frame: 输入帧
            metrics: 性能指标数据

        Returns:
            ndarray: 处理后的帧
        """
        if not self.is_enabled() or not self._metrics_enabled:
            return frame

        try:
            # 更新系统统计
            self._system_summary['frames_processed'] += 1

            # 记录性能指标
            self._record_metrics(metrics)

            # 检查是否需要周期性记录
            current_time = time.time()
            if current_time - self._last_record_time >= self._record_interval:
                self._last_record_time = current_time
                self._record_periodic_data()

            # 检查是否需要生成快照
            if self._snapshot_enabled and current_time - self._last_snapshot_time >= self._snapshot_interval:
                self._last_snapshot_time = current_time
                self._take_snapshot()

            # 检测异常
            if self._anomaly_detection:
                self._detect_anomalies(metrics)

            # 应用可视化
            if self._visualization_enabled and self._overlay_metrics:
                frame = self._overlay_metrics_on_frame(frame, metrics)

            return frame

        except Exception as e:
            logger.error(f"处理帧时出错: {e}")
            return frame

    def generate_report(self, final=False):
        """
        生成调试报告

        Args:
            final: 是否为最终报告

        Returns:
            str: 报告文件路径
        """
        try:
            # 创建报告文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            report_type = 'final' if final else 'interim'
            report_file = os.path.join(self._report_dir,
                                       f"debug_report_{report_type}_{timestamp}.json")

            # 收集报告数据
            report_data = self._collect_report_data(final)

            # 保存报告
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"调试报告已生成: {report_file}")

            # 发布报告生成事件
            if self._event_system and hasattr(self._event_system, 'publish'):
                self._event_system.publish(
                    "debug_report_generated",
                    {
                        'plugin_id': self._id,
                        'report_file': report_file,
                        'report_type': report_type
                    }
                )

            return report_file

        except Exception as e:
            logger.error(f"生成调试报告时出错: {e}")
            return None

    def analyze_performance(self, component_id=None):
        """
        分析系统或组件性能

        Args:
            component_id: 特定组件ID，如果为None则分析整个系统

        Returns:
            Dict: 性能分析结果
        """
        try:
            analysis = {}

            # 整体性能
            if component_id is None:
                # FPS 统计
                if self._fps_history:
                    avg_fps = sum(self._fps_history) / len(self._fps_history)
                    min_fps = min(self._fps_history)
                    max_fps = max(self._fps_history)
                    analysis['fps'] = {
                        'average': avg_fps,
                        'min': min_fps,
                        'max': max_fps,
                        'below_threshold_percent': sum(
                            1 for fps in self._fps_history if
                            fps < self._fps_threshold) / len(
                            self._fps_history) * 100
                    }

                # 延迟统计
                if self._frame_times:
                    avg_latency = sum(self._frame_times) / len(
                        self._frame_times) * 1000
                    max_latency = max(self._frame_times) * 1000
                    analysis['latency'] = {
                        'average_ms': avg_latency,
                        'max_ms': max_latency,
                        'above_threshold_percent': sum(
                            1 for t in self._frame_times if
                            t * 1000 > self._latency_threshold) / len(
                            self._frame_times) * 100
                    }

                # 各组件时间占比
                total_time = 0
                component_times = {}

                if self._detection_times:
                    avg_detection = sum(self._detection_times) / len(
                        self._detection_times)
                    component_times['detection'] = avg_detection
                    total_time += avg_detection

                if self._mapping_times:
                    avg_mapping = sum(self._mapping_times) / len(
                        self._mapping_times)
                    component_times['mapping'] = avg_mapping
                    total_time += avg_mapping

                if self._recognition_times:
                    avg_recognition = sum(self._recognition_times) / len(
                        self._recognition_times)
                    component_times['recognition'] = avg_recognition
                    total_time += avg_recognition

                if total_time > 0:
                    analysis['time_distribution'] = {
                        component: {
                            'time': time,
                            'percentage': (time / total_time) * 100
                        } for component, time in component_times.items()
                    }

                # 瓶颈分析
                if self._bottleneck_analysis and total_time > 0:
                    bottlenecks = []
                    for component, time in component_times.items():
                        if time / total_time > 0.4:  # 如果某个组件占用超过40%的时间
                            bottlenecks.append({
                                'component': component,
                                'time': time,
                                'percentage': (time / total_time) * 100
                            })

                    analysis['bottlenecks'] = bottlenecks

                # 稳定性分析
                if self._frame_times and len(self._frame_times) > 10:
                    # 计算变异系数(标准差/均值)，表示稳定性
                    mean = sum(self._frame_times) / len(self._frame_times)
                    variance = sum(
                        (t - mean) ** 2 for t in self._frame_times) / len(
                        self._frame_times)
                    std_dev = variance ** 0.5
                    cv = std_dev / mean if mean > 0 else 0

                    analysis['stability'] = {
                        'coefficient_of_variation': cv,
                        'std_deviation': std_dev,
                        'rating': 'good' if cv < 0.2 else 'medium' if cv < 0.5 else 'poor'
                    }

            # 特定组件分析
            else:
                # 查找特定组件的状态和性能数据
                if component_id in self._component_states:
                    component_data = self._component_states[component_id]
                    analysis['component'] = {
                        'id': component_id,
                        'state': component_data.get('state', 'unknown'),
                        'time': component_data.get('time', 0),
                        'error_count': component_data.get('error_count', 0)
                    }

            # 添加系统状态摘要
            analysis['system_summary'] = {
                'total_frames': self._system_summary['frames_processed'],
                'total_detections': self._system_summary['detections'],
                'total_recognitions': self._system_summary[
                    'actions_recognized'],
                'error_count': len(self._system_summary['errors']),
                'warning_count': len(self._system_summary['warnings'])
            }

            return analysis

        except Exception as e:
            logger.error(f"分析性能时出错: {e}")
            return {'error': str(e)}

    def get_debug_info(self):
        """
        获取调试信息

        Returns:
            dict: 调试信息
        """
        debug_info = {
            'plugin_id': self._id,
            'enabled': self._enabled,
            'metrics_enabled': self._metrics_enabled,
            'fps': self._fps_history[-1] if self._fps_history else 0,
            'frames_processed': self._system_summary['frames_processed'],
            'detections': self._system_summary['detections'],
            'actions_recognized': self._system_summary['actions_recognized'],
            'errors': len(self._system_summary['errors']),
            'warnings': len(self._system_summary['warnings'])
        }

        # 添加性能指标
        if self._metrics_enabled and self._frame_times:
            avg_latency = sum(self._frame_times) / len(self._frame_times) * 1000
            debug_info['avg_latency_ms'] = avg_latency

        return debug_info

    # ============= 辅助方法 =============

    def _setup_event_subscriptions(self):
        """设置事件订阅"""
        try:
            if not self._event_system:
                return

            # 订阅FPS更新事件
            if hasattr(self._event_system, 'subscribe'):
                self._event_system.subscribe("fps_updated",
                                             self._on_fps_updated)
                self._event_system.subscribe("person_detected",
                                             self._on_person_detected)
                self._event_system.subscribe("action_recognized",
                                             self._on_action_recognized)
                self._event_system.subscribe("position_mapped",
                                             self._on_position_mapped)
                self._event_system.subscribe("system_state_changed",
                                             self._on_system_state_changed)
                self._event_system.subscribe("error_occurred",
                                             self._on_error_occurred)

                logger.info("已设置事件订阅")
        except Exception as e:
            logger.error(f"设置事件订阅时出错: {e}")

    def _record_system_config(self, context):
        """记录系统配置"""
        try:
            if not context:
                return

            config = {}

            # 记录组件列表
            if 'components' in context:
                config['components'] = []
                for comp_id, comp in context['components'].items():
                    comp_info = {
                        'id': comp_id,
                        'type': getattr(comp, '_component_type', 'unknown'),
                        'initialized': hasattr(comp,
                                               'is_initialized') and comp.is_initialized()
                    }
                    config['components'].append(comp_info)

            # 记录插件列表
            if 'plugins' in context:
                config['plugins'] = []
                for plugin_id, plugin in context['plugins'].items():
                    plugin_info = {
                        'id': plugin_id,
                        'type': plugin.__class__.__name__ if hasattr(plugin,
                                                                     '__class__') else 'unknown',
                        'initialized': hasattr(plugin,
                                               'is_initialized') and plugin.is_initialized(),
                        'enabled': hasattr(plugin,
                                           'is_enabled') and plugin.is_enabled()
                    }
                    config['plugins'].append(plugin_info)

            # 记录相机信息
            if 'camera' in context:
                camera = context['camera']
                config['camera'] = {
                    'width': getattr(camera, 'width', 0),
                    'height': getattr(camera, 'height', 0),
                    'fps': getattr(camera, 'fps', 0)
                }

            # 保存配置
            self._system_summary['config'] = config

        except Exception as e:
            logger.error(f"记录系统配置时出错: {e}")

    def _record_metrics(self, metrics):
        """
        记录性能指标

        Args:
            metrics: 性能指标数据
        """
        try:
            # 提取指标
            if 'frame_time' in metrics:
                self._frame_times.append(metrics['frame_time'])

            if 'fps' in metrics:
                self._fps_history.append(metrics['fps'])

            if 'detection_time' in metrics:
                self._detection_times.append(metrics['detection_time'])

            if 'mapping_time' in metrics:
                self._mapping_times.append(metrics['mapping_time'])

            if 'recognition_time' in metrics:
                self._recognition_times.append(metrics['recognition_time'])

            if 'memory' in metrics:
                self._memory_usage.append(metrics['memory'])

            # 更新组件状态
            if 'components' in metrics:
                for comp_id, comp_data in metrics['components'].items():
                    if comp_id not in self._component_states:
                        self._component_states[comp_id] = {}

                    self._component_states[comp_id].update(comp_data)

        except Exception as e:
            logger.error(f"记录指标时出错: {e}")

    def _record_periodic_data(self):
        """记录周期性数据"""
        try:
            # 记录当前内存使用
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info
