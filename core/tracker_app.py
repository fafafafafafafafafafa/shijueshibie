# -*- coding: utf-8 -*-
import cv2
import time
import logging
import numpy as np
import queue
import threading

# 在TrackerApp.__init__方法中添加
from calibration.calibration import CalibrationManager
from ui.display_manager import DisplayManager
from ui.input_handler import InputHandler
from .SimplifiedSystemManager import SimplifiedSystemManager
from .AsyncComponents import AsyncPipeline
from utils.cache_monitor import get_monitor
from utils.thread_utils import shutdown_thread_pool
from utils.event_system import EnhancedEventSystem as EventSystem
from utils.event_types import EventTypes
from utils.config import AppConfig
from utils.event_logger import EventLogger  # 仅用于类型提示，如果使用
from utils.logger_config import setup_logger, init_root_logger, setup_utf8_console
from interface.detector_interface import PersonDetectorInterface
from interface.action_recognizer_interface import ActionRecognizerInterface
from interface.position_mapper_interface import PositionMapperInterface
from utils.conditional_events import get_conditional_event_system, where, Condition
# 添加以下代码来初始化事件系统
from utils.event_system import get_event_system
from utils.ui_events import UIEventPublisher
from ui.enhanced_display_manager import EnhancedDisplayManager
from ui.enhanced_input_handler import EnhancedInputHandler

logger = setup_logger(__name__)


class TrackerApp:
    """主应用类，支持使用简化检测器的模块化架构，并具有异步处理能力"""

    def __init__(self, detector, action_recognizer, position_mapper,
                 visualizer, system_manager=None,config=None,event_logger=None):
        """
                初始化TrackerApp

                Args:
                    detector (PersonDetectorInterface): 人体检测器
                    action_recognizer (ActionRecognizerInterface): 动作识别器
                    position_mapper (PositionMapperInterface): 位置映射器
                    visualizer: 可视化器
                    system_manager: 可选的系统管理器
                    config (AppConfig): 可选的应用配置
        """
        # 保存事件日志记录器
        self.event_logger = event_logger
        # 配置日志记录
        self.logger = logger

        # 获取事件系统
        self.event_system = get_event_system(
            history_capacity=100,  # 保存最近100个事件的历史
            enable_logging=True,  # 启用事件日志
            async_mode=True  # 使用异步事件处理
        )

        # 创建事件发布器
        self.publisher = UIEventPublisher("tracker_app")

        # 验证接口
        if not isinstance(detector, PersonDetectorInterface):
            raise TypeError("detector 必须实现 PersonDetectorInterface")
        if not isinstance(action_recognizer, ActionRecognizerInterface):
            raise TypeError(
                "action_recognizer 必须实现 ActionRecognizerInterface")
        if not isinstance(position_mapper, PositionMapperInterface):
            raise TypeError("position_mapper 必须实现 PositionMapperInterface")

        # 主要组件
        self.detector = detector  # 可以是SimplifiedPersonDetector或原始PersonDetector
        self.action_recognizer = action_recognizer
        self.position_mapper = position_mapper
        self.visualizer = visualizer or self._create_default_visualizer()

        # 配置
        self.config = config or AppConfig()

        # 初始化事件系统
        self.events = EventSystem()

        # 注册事件
        self._setup_events()

        # 异步管道相关 - 在feature toggles之前初始化
        self.async_pipeline = None
        self.use_async = False  # 默认使用同步处理模式

        # 初始化相机
        self.camera = None
        self.frame_width = 0
        self.frame_height = 0
        self.init_camera()


        # 系统管理器 - 整合了状态管理、性能监控和资源监控功能
        if system_manager:
            self.system_manager = system_manager
            # 如果系统管理器没有事件系统，则设置一个
            if hasattr(self.system_manager, 'set_event_system'):
                self.system_manager.set_event_system(self.events)
            elif not hasattr(self.system_manager, 'events'):
                self.system_manager.events = self.events
        else:
            # 创建默认的系统管理器
            from .SimplifiedSystemManager import SimplifiedSystemManager
            self.system_manager = SimplifiedSystemManager(
                log_interval=self.config.log_interval,
                memory_warning_threshold=self.config.memory_warning_threshold,
                memory_critical_threshold=self.config.memory_critical_threshold,
                event_system=self.events  # 传递事件系统
            )

        # 注册各组件的缓存
        cache_monitor = get_monitor()
        if hasattr(self.detector, 'detection_cache'):
            cache_monitor.register_cache('detector',
                                             self.detector.detection_cache)
        if hasattr(self.action_recognizer, 'action_cache'):
            cache_monitor.register_cache('action_recognizer',
                                             self.action_recognizer.action_cache)
        if hasattr(self.position_mapper, 'position_cache'):
            cache_monitor.register_cache('position_mapper',
                                             self.position_mapper.position_cache)
        # 辅助组件
        self.calibration_manager = CalibrationManager(detector, position_mapper)
        self.display_manager = EnhancedDisplayManager(self.visualizer)
        self.input_handler = EnhancedInputHandler(self.display_manager)
        self._setup_input_handlers()

        self.register_all_caches()

        # 注册输入处理函数
        self.input_handler.register_handler('r', self.recalibrate, '重新校准')
        self.input_handler.register_handler('h', self.show_help, '显示帮助')
        # 注册异步模式切换处理
        self.input_handler.register_handler('y', self.toggle_async_mode,
                                            '切换异步模式')


        # 设置功能切换
        self.setup_feature_toggles()

        # 运行状态
        self.frame_count = 0
        self.last_valid_person = None
        self.last_valid_action = "Static"
        self.running = False
        self.last_error_time = 0
        self.error_cooldown = 5.0  # 错误冷却时间（秒）

        self.logger.info("模块化TrackerApp初始化完成")

        # 最后添加：启动缓存监控
        monitor = get_monitor()
        monitor.start_monitoring()
        self.logger.info("缓存监控已启动")

        # 立即持久化当前统计
        stats = monitor.get_all_stats()
        for name in stats:
            monitor._persist_stats(name)

        self.logger.info("缓存监控已启动")
        # 修改CalibrationManager的初始化
        self.calibration_manager = CalibrationManager(detector, position_mapper,
                                                      self.events)

        # 升级为条件性事件系统
        self.conditional_events = get_conditional_event_system(self.events)
        # 设置条件性事件处理器
        self._setup_conditional_events()
        self._setup_feature_callbacks()
        # 新增: 设置事件处理器
        self._setup_event_handlers()

    def _toggle_mediapipe(self, feature_name, state):
        """MediaPipe功能切换回调"""
        try:
            print(f"切换MediaPipe功能: {state}")

            # 这里添加调整detector的代码，例如：
            if hasattr(self.detector, 'set_mediapipe_mode'):
                self.detector.set_mediapipe_mode(state)

            # 发布通知
            self.publisher.publish_notification(
                f"MediaPipe功能已{'启用' if state else '禁用'}",
                level="info"
            )

            return True
        except Exception as e:
            print(f"切换MediaPipe功能失败: {e}")

            # 发布错误通知
            self.publisher.publish_notification(
                f"切换MediaPipe功能失败: {e}",
                level="error"
            )

            return False

    def _toggle_ml_model(self, feature_name, state):
        """机器学习模型功能切换回调"""
        try:
            print(f"切换机器学习模型功能: {state}")

            # 这里添加启用/禁用ML模型的代码
            if hasattr(self.action_recognizer, 'use_ml_model'):
                self.action_recognizer.use_ml_model = state

            # 发布通知
            self.publisher.publish_notification(
                f"机器学习模型已{'启用' if state else '禁用'}",
                level="info"
            )

            return True
        except Exception as e:
            print(f"切换机器学习模型功能失败: {e}")

            # 发布错误通知
            self.publisher.publish_notification(
                f"切换机器学习模型功能失败: {e}",
                level="error"
            )

            return False

    def _toggle_dtw(self, feature_name, state):
        """DTW功能切换回调"""
        try:
            print(f"切换DTW功能: {state}")

            # 这里添加启用/禁用DTW的代码
            if hasattr(self.action_recognizer, 'use_dtw'):
                self.action_recognizer.use_dtw = state

            # 发布通知
            self.publisher.publish_notification(
                f"DTW功能已{'启用' if state else '禁用'}",
                level="info"
            )

            return True
        except Exception as e:
            print(f"切换DTW功能失败: {e}")

            # 发布错误通知
            self.publisher.publish_notification(
                f"切换DTW功能失败: {e}",
                level="error"
            )

            return False

    def _toggle_threading(self, feature_name, state):
        """多线程处理功能切换回调"""
        try:
            print(f"切换多线程处理功能: {state}")

            # 这里添加启用/禁用多线程的代码
            self.use_threading = state

            # 发布通知
            self.publisher.publish_notification(
                f"多线程处理已{'启用' if state else '禁用'}",
                level="info"
            )

            return True
        except Exception as e:
            print(f"切换多线程处理功能失败: {e}")

            # 发布错误通知
            self.publisher.publish_notification(
                f"切换多线程处理功能失败: {e}",
                level="error"
            )

            return False

    def _toggle_async(self, feature_name, state):
        """异步处理功能切换回调"""
        try:
            print(f"切换异步处理功能: {state}")

            # 这里添加启用/禁用异步处理的代码
            self.use_async = state

            # 发布通知
            self.publisher.publish_notification(
                f"异步处理已{'启用' if state else '禁用'}",
                level="info"
            )

            return True
        except Exception as e:
            print(f"切换异步处理功能失败: {e}")

            # 发布错误通知
            self.publisher.publish_notification(
                f"切换异步处理功能失败: {e}",
                level="error"
            )

            return False


    def _setup_feature_callbacks(self):
        """设置功能切换回调函数"""
        # 注册MediaPipe功能切换回调
        self.input_handler.register_feature_toggle('m', self._toggle_mediapipe)

        # 注册机器学习模型功能切换回调
        self.input_handler.register_feature_toggle('l', self._toggle_ml_model)

        # 注册DTW功能切换回调
        self.input_handler.register_feature_toggle('w', self._toggle_dtw)

        # 注册多线程处理功能切换回调
        self.input_handler.register_feature_toggle('p', self._toggle_threading)

        # 注册异步处理功能切换回调
        self.input_handler.register_feature_toggle('y', self._toggle_async)

    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 订阅人体检测事件
        self.event_system.subscribe(
            "person_detected",
            self._handle_person_detected
        )

        # 订阅动作识别事件
        self.event_system.subscribe(
            "action_recognized",
            self._handle_action_recognized
        )

        # 订阅位置映射事件
        self.event_system.subscribe(
            "position_mapped",
            self._handle_position_mapped
        )

    def _handle_person_detected(self, data):
        """处理人体检测事件"""
        if 'person' in data:
            person = data['person']
            # 这里可以添加额外的处理逻辑
            print(f"检测到人体: 位置={person.get('bbox', 'unknown')}")

    def _handle_action_recognized(self, data):
        """处理动作识别事件"""
        if 'action' in data:
            action = data['action']
            # 这里可以添加额外的处理逻辑
            print(f"识别到动作: {action}")

    def _handle_position_mapped(self, data):
        """处理位置映射事件"""
        if 'position' in data:
            position = data['position']
            depth = data.get('depth', None)
            # 这里可以添加额外的处理逻辑
            print(f"映射位置: {position}, 深度: {depth}")


    def _setup_conditional_events(self):
        """设置条件性事件处理"""

        # 示例1: 监听高速移动
        self.conditional_events.subscribe_if(
            "position_mapped",
            where('velocity').greater_than(50).AND().equals('action', 'Moving'),
            self._handle_fast_movement
        )

        # 示例2: 监听低置信度检测
        self.conditional_events.subscribe_if(
            "person_detected",
            where('person.confidence').less_than(0.6),
            self._handle_low_confidence
        )

        # 示例3: 监听资源警告
        self.conditional_events.subscribe_when(
            lambda: where('memory_percent').greater_than(
                self.config.memory_warning_threshold)
        ).then(
            "resource_warning",
            self._handle_resource_warning,
            priority=10  # 高优先级
        )

    def _handle_fast_movement(self, data):
        """处理高速移动"""
        self.logger.info(f"检测到高速移动: {data.get('velocity', 0)}")
        # 处理逻辑...

    def _handle_low_confidence(self, data):
        """处理低置信度检测"""
        self.logger.warning(
            f"低置信度检测: {data.get('person', {}).get('confidence', 0)}")
        # 处理逻辑...

    def _handle_resource_warning(self, data):
        """处理资源警告"""
        self.logger.warning(
            f"资源警告: 内存使用率 {data.get('memory_percent', 0)}%")
        # 可能需要调整性能设置
        if self.async_pipeline:
            self.async_pipeline.set_performance_mode('high_speed')

    def _setup_events(self):
        """设置事件订阅"""
        # 订阅各种事件
        self.events.subscribe("person_detected", self._on_person_detected)
        self.events.subscribe("frame_processed", self._on_frame_processed)
        self.events.subscribe("system_state_changed",
                              self._on_system_state_changed)
        self.events.subscribe("key_pressed", self._on_key_pressed)
        self.events.subscribe("feature_toggled", self._on_feature_toggled)
        self.events.subscribe("calibration_completed",
                              self._on_calibration_completed)

        # 添加资源监控事件订阅
        self.events.subscribe("resource_state_changed",
                              self._on_resource_state_changed)

        # 添加Y键特殊事件处理 - 用于异步模式切换
        self.events.subscribe("key_y_pressed", self._on_y_key_pressed)

        # 将事件系统传递给输入处理器
        if hasattr(self, 'input_handler') and self.input_handler is not None:
            if hasattr(self.input_handler, 'set_event_system'):
                self.input_handler.set_event_system(self.events)
            else:
                # 直接设置属性
                self.input_handler.events = self.events

    def _on_y_key_pressed(self, data=None):
        """处理Y键按下事件 - 切换异步模式"""
        self.logger.info("收到Y键事件，切换异步模式")
        # 调用切换异步模式方法
        self.toggle_async_mode()

    # 添加资源状态变化事件处理方法
    def _on_resource_state_changed(self, data):
        """处理资源状态变化事件"""
        old_level = data.get('old_level')
        new_level = data.get('new_level')
        memory_usage = data.get('memory_usage')
        cpu_usage = data.get('cpu_usage')

        level_names = ["正常", "警告", "临界"]
        self.logger.warning(
            f"资源状态从{level_names[old_level]}变为{level_names[new_level]}")
        self.logger.warning(
            f"内存使用率: {memory_usage:.1f}%, CPU使用率: {cpu_usage:.1f}%")

        # 如果资源状态达到临界，考虑采取措施
        if new_level == 2:  # 临界状态
            # 如果不是异步模式，可以建议切换
            if not self.use_async:
                print("系统资源紧张，推荐使用异步处理模式（按'y'切换）")
            # 禁用一些高资源消耗的功能
            if self.display_manager.feature_states.get('ml_model', False):
                print("系统资源紧张，建议禁用ML模型（按'l'切换）")

    def _setup_input_handlers(self):
        """设置输入处理函数"""
        # 注册输入处理函数
        self.input_handler.register_handler('r', self.recalibrate, '重新校准')
        self.input_handler.register_handler('h', self.input_handler.show_help,
                                            '显示帮助')
        self.input_handler.register_handler('y', self.toggle_async_mode,
                                            '切换异步模式')

        # 注册功能切换回调
        self.input_handler.register_feature_toggle('m', self.toggle_feature)
        self.input_handler.register_feature_toggle('l', self.toggle_feature)
        self.input_handler.register_feature_toggle('w', self.toggle_feature)
        self.input_handler.register_feature_toggle('p', self.toggle_feature)

    def _on_person_detected(self, person):
        """处理人体检测事件"""
        try:
            # 识别动作
            action = self.action_recognizer.recognize_action(person)

            # 发布动作识别事件
            self.events.publish("action_recognized", {
                'person': person,
                'action': action
            })
        except Exception as e:
            self.logger.error(f"动作识别错误: {e}")

    def _on_frame_processed(self, data):
        """处理帧处理完成事件"""
        # 更新显示
        self.display_manager.display_frames(data['frame_viz'], data['room_viz'])

    def _on_system_state_changed(self, state):
        """处理系统状态变化事件"""
        self.logger.info(f"系统状态变化: {state}")

    def _on_key_pressed(self, key):
        """处理按键事件"""
        # 这里可以处理特定的按键事件
        pass

    def setup_feature_toggles(self):
        """设置功能切换处理"""
        # 注册功能切换回调
        self.input_handler.register_feature_toggle('m', self.toggle_feature)
        self.input_handler.register_feature_toggle('l', self.toggle_feature)
        self.input_handler.register_feature_toggle('w', self.toggle_feature)
        self.input_handler.register_feature_toggle('p', self.toggle_feature)
        #self.input_handler.register_feature_toggle('y', self.toggle_feature)

        # 初始化显示状态
        self.update_feature_display_states()

        self.logger.info("功能切换处理已设置")

    def update_feature_display_states(self):
        """更新功能显示状态"""
        # 检查各功能状态并更新DisplayManager中的状态显示

        # 检查MediaPipe状态 - 兼容SimplifiedPersonDetector和旧版检测器
        if hasattr(self.detector, 'using_mediapipe'):
            self.display_manager.update_feature_state('mediapipe',
                                                      self.detector.using_mediapipe)

        # 检查ML模型状态
        if hasattr(self.action_recognizer, 'ml_recognizer'):
            self.display_manager.update_feature_state('ml_model',
                                                      self.action_recognizer.ml_recognizer is not None)

        # 检查DTW状态
        if hasattr(self.action_recognizer, 'dtw_recognizer'):
            self.display_manager.update_feature_state('dtw',
                                                      self.action_recognizer.dtw_recognizer is not None)

        # 检查多线程状态
        if hasattr(self.action_recognizer, 'config'):
            self.display_manager.update_feature_state('threading',
                                                      self.action_recognizer.config.get(
                                                          'enable_threading',
                                                          False))

        # 更新异步管道状态
        self.display_manager.update_feature_state('async', self.use_async)

    def toggle_feature(self, feature_name, new_state):
        """
        切换功能开关

        使用接口方法切换功能，并通过事件系统通知状态变更

        Args:
            feature_name: 功能名称
            new_state: 新状态

        Returns:
            bool: 是否成功切换
        """
        self.logger.info(
            f"尝试{'启用' if new_state else '禁用'}功能: {feature_name}")

        # 定义功能所属组件的映射
        feature_component_map = {
            # 检测器功能
            'mediapipe': self.detector,

            # 动作识别器功能
            'ml_model': self.action_recognizer,
            'dtw': self.action_recognizer,
            'threading': self.action_recognizer,

            # 位置映射器功能
            'backward_enhancement': self.position_mapper,
            'simple_mapping': self.position_mapper,
            'debug_mode': self.position_mapper,

            # 异步处理功能 - 特殊处理
            'async': None
        }

        # 特殊处理异步模式
        if feature_name == 'async':
            success = self.set_async_mode(new_state)
            if success:
                self.events.publish("feature_toggled", {
                    'feature_name': feature_name,
                    'state': new_state,
                    'component': 'tracker_app'
                })
            return success

        # 获取对应的组件
        component = feature_component_map.get(feature_name)
        if not component:
            self.logger.error(f"未知功能: {feature_name}")
            return False

        try:
            # 使用接口方法切换功能
            if hasattr(component, 'toggle_feature'):
                success = component.toggle_feature(feature_name, new_state)

                # 如果切换成功，发布事件
                if success:
                    self.events.publish("feature_toggled", {
                        'feature_name': feature_name,
                        'state': new_state,
                        'component': component.__class__.__name__
                    })

                    # 更新UI显示状态
                    self.display_manager.update_feature_state(feature_name,
                                                              new_state)
                else:
                    self.events.publish("feature_toggle_failed", {
                        'feature_name': feature_name,
                        'state': new_state,
                        'reason': "组件返回失败"
                    })

                return success
            else:
                # 组件不支持标准接口方法，尝试使用旧版本的方式
                self.logger.warning(
                    f"组件 {component.__class__.__name__} 不支持标准接口方法")

                # 旧版本兼容处理 - 检测器
                if component == self.detector and feature_name == 'mediapipe':
                    if hasattr(component, 'using_mediapipe'):
                        component.using_mediapipe = new_state
                        if new_state and not hasattr(component, 'pose'):
                            if hasattr(component, '_init_mediapipe'):
                                component._init_mediapipe()
                            elif hasattr(component, 'init_mediapipe'):
                                component.init_mediapipe()

                        self.events.publish("feature_toggled", {
                            'feature_name': feature_name,
                            'state': new_state,
                            'component': 'detector',
                            'legacy': True
                        })

                        self.display_manager.update_feature_state(feature_name,
                                                                  new_state)
                        return True

                # 旧版本兼容处理 - 动作识别器
                elif component == self.action_recognizer and feature_name in [
                    'ml_model', 'dtw', 'threading']:
                    if hasattr(component, 'config'):
                        # 动作识别器的旧版本处理逻辑...
                        # 根据feature_name执行相应的处理
                        if feature_name == 'ml_model':
                            component.config['enable_ml_model'] = new_state
                            # 应用更改
                            if new_state:
                                if not hasattr(component,
                                               'ml_recognizer') or component.ml_recognizer is None:
                                    try:
                                        from advanced_recognition import \
                                            MLActionRecognizer
                                        component.ml_recognizer = MLActionRecognizer(
                                            component.config)
                                    except ImportError:
                                        if hasattr(component,
                                                   'MLActionRecognizer'):
                                            component.ml_recognizer = component.MLActionRecognizer(
                                                component.config)
                                        else:
                                            self.logger.error(
                                                "无法导入MLActionRecognizer")
                                            return False
                            else:
                                component.ml_recognizer = None
                        elif feature_name == 'dtw':
                            component.config['enable_dtw'] = new_state
                            # 应用更改
                            if new_state:
                                if not hasattr(component,
                                               'dtw_recognizer') or component.dtw_recognizer is None:
                                    try:
                                        from advanced_recognition import \
                                            DTWActionRecognizer
                                        component.dtw_recognizer = DTWActionRecognizer(
                                            component.config)
                                    except ImportError:
                                        if hasattr(component,
                                                   'DTWActionRecognizer'):
                                            component.dtw_recognizer = component.DTWActionRecognizer(
                                                component.config)
                                        else:
                                            self.logger.error(
                                                "无法导入DTWActionRecognizer")
                                            return False
                            else:
                                component.dtw_recognizer = None
                        elif feature_name == 'threading':
                            # 记录当前值以便恢复
                            old_value = component.config.get('enable_threading',
                                                             False)

                            # 设置新值
                            component.config['enable_threading'] = new_state

                            # 如果开启多线程，确保线程池可用
                            if new_state:
                                try:
                                    from utils import get_thread_pool
                                    thread_pool = get_thread_pool(
                                        component.config.get('max_workers', 2))
                                except Exception as e:
                                    self.logger.error(f"初始化线程池失败: {e}")
                                    # 恢复旧值
                                    component.config[
                                        'enable_threading'] = old_value
                                    return False

                        # 发布事件和更新UI
                        self.events.publish("feature_toggled", {
                            'feature_name': feature_name,
                            'state': new_state,
                            'component': 'action_recognizer',
                            'legacy': True
                        })

                        self.display_manager.update_feature_state(feature_name,
                                                                  new_state)
                        return True

                # 如果没有适用的旧版本处理，返回失败
                self.logger.error(
                    f"无法切换功能 {feature_name}: 不支持的接口和旧版本")
                return False

        except Exception as e:
            self.logger.error(f"切换功能 {feature_name} 时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # 发布错误事件
            self.events.publish("feature_toggle_error", {
                'feature_name': feature_name,
                'state': new_state,
                'error': str(e)
            })

            return False

    def init_camera(self):
        """初始化相机"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("无法打开相机")

            self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(
                f"相机初始化成功: {self.frame_width}x{self.frame_height}")

        except Exception as e:
            self.logger.error(f"相机初始化失败: {e}")
            raise

    def recalibrate(self):
        """重新校准系统"""
        if self.camera is not None and self.camera.isOpened():
            self.logger.info("开始重新校准")
            self.calibration_manager.calibrate(self.camera)
        else:
            self.logger.error("无法重新校准 - 相机未打开")

    def show_help(self):
        """显示帮助信息"""
        self.input_handler.show_help()

    def toggle_async_mode(self):
        """切换异步/同步处理模式"""
        self.logger.info("切换异步处理模式")
        print("\n尝试切换异步模式...")

        # 获取当前状态并计算目标状态
        new_state = not self.use_async
        print(
            f"当前状态: {'异步' if self.use_async else '同步'}, 准备切换到: {'异步' if new_state else '同步'}")

        # 调用设置方法
        success = self.set_async_mode(new_state)

        if success:
            mode_name = "异步" if new_state else "同步"
            print(f"已成功切换到{mode_name}处理模式")

            # 发布功能切换事件
            if hasattr(self, 'events') and self.events:
                self.events.publish("feature_toggled", {
                    'feature_name': 'async',
                    'state': new_state,
                    'component': 'tracker_app'
                })
        else:
            print(f"切换到{'异步' if new_state else '同步'}模式失败")

        return success

    def set_async_mode(self, enabled):
        """设置异步模式

        Args:
            enabled: 是否启用异步模式

        Returns:
            bool: 是否成功设置
        """
        # 如果当前状态与目标状态相同，不进行操作
        if self.use_async == enabled:
            return True

        try:
            if enabled:
                # 切换到异步模式
                if not self.async_pipeline:
                    # 创建异步管道
                    print("创建异步管道...")
                    self.async_pipeline = self.create_async_pipeline()
                    if not self.async_pipeline:
                        print("异步管道创建失败")
                        return False

                # 如果应用正在运行，启动异步管道
                if self.running:
                    print("启动异步管道...")
                    self.async_pipeline.start()

                self.use_async = True
                self.logger.info("已切换到异步处理模式")
                print("已切换到异步处理模式")
            else:
                # 切换到同步模式
                if self.async_pipeline and self.running:
                    print("停止异步管道...")
                    self.async_pipeline.stop()

                self.use_async = False
                self.logger.info("已切换到同步处理模式")
                print("已切换到同步处理模式")

            # 更新显示状态
            self.display_manager.update_feature_state('async', self.use_async)
            return True

        except Exception as e:
            self.logger.error(f"切换异步模式失败: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            self.logger.error(traceback_str)
            print(f"切换异步模式失败: {e}")
            print(traceback_str)
            return False

    def process_detected_person(self, person, frame, current_time):
        """处理新检测到的人

        Args:
            person: 人体检测数据
            frame: 当前帧
            current_time: 当前时间戳

        Returns:
            tuple: 处理后的帧和房间视图
        """
        # 保存最后有效检测
        self.last_valid_person = person

        # 添加必要信息
        person['calibration_height'] = self.position_mapper.calibration_height
        person['detection_time'] = current_time

        # 发布人体准备好识别动作事件
        self.events.publish("person_ready_for_action", person)

        # 识别动作
        action_start = time.time()
        try:
            action = self.action_recognizer.recognize_action(person)
            self.last_valid_action = action

            # 发布动作识别事件
            self.events.publish("action_recognized", {
                'person': person,
                'action': action
            })
        except Exception as e:
            self.logger.error(f"动作识别错误: {e}")
            action = self.last_valid_action
        action_time = time.time() - action_start

        # 映射位置
        mapping_start = time.time()
        try:
            # 映射到房间坐标
            room_x, room_y, depth = self.position_mapper.map_position_to_room(
                self.frame_width, self.frame_height,
                self.visualizer.room_width,
                self.visualizer.room_height,
                person
            )

            # 获取稳定的位置
            stable_x, stable_y = self.position_mapper.get_stable_position(
                room_x, room_y, action)

            # 平滑位置
            smooth_x, smooth_y = self.position_mapper.smooth_position(
                stable_x, stable_y)

            # 添加轨迹点
            self.visualizer.add_trail_point(smooth_x, smooth_y)

            # 发布位置映射事件
            self.events.publish("position_mapped", {
                'position': (smooth_x, smooth_y),
                'depth': depth,
                'action': action
            })

            # 可视化
            frame_viz = self.visualizer.visualize_frame(frame,
                                                        person,
                                                        action,
                                                        self.detector)
            room_viz = self.visualizer.visualize_room(
                (smooth_x, smooth_y),
                depth, action)
            self.display_manager.last_room_viz = room_viz

        except Exception as e:
            self.logger.error(f"位置映射错误: {e}")
            frame_viz = self.visualizer.visualize_frame(frame,
                                                        person,
                                                        action,
                                                        self.detector)
            room_viz = self.display_manager.get_last_room_viz()

        mapping_time = time.time() - mapping_start

        return frame_viz, room_viz

    def run(self):
        """运行追踪应用"""
        # 首先进行校准
        if not self.calibration_manager.calibrate(self.camera):
            self.logger.warning("校准未完成，使用默认值")

        self.running = True

        # 启动异步管道（如果启用）
        if self.use_async and self.async_pipeline:
            self.async_pipeline.start()

        self.logger.info(
            "追踪器开始运行. 按 'q' 退出, 'd' 切换调试信息, 'r' 重新校准, 'y' 切换异步模式.")
        print(
            "追踪器运行中. 按 'q' 退出, 'd' 切换调试信息, 'r' 重新校准, 'h' 显示帮助.")
        print(
            "功能切换键: 'm'-MediaPipe, 'l'-ML模型, 'w'-DTW, 'p'-多线程, 'y'-异步模式")

        try:
            if self.use_async:
                self.async_main_loop()
            else:
                self.sync_main_loop()
        except KeyboardInterrupt:
            self.logger.info("用户终止程序")
            print("\n用户终止程序")
        except Exception as e:
            self.logger.error(f"程序运行时发生错误: {e}", exc_info=True)
            print(f"程序运行时发生错误: {e}")
        finally:
            self.cleanup()

    def sync_main_loop(self):
        """同步处理主循环"""
        recovery_counter = 0
        memory_check_counter = 0  # 用于资源检查计数
        feature_update_counter = 0  # 用于功能状态更新计数
        fps_counter = 0
        fps_timer = time.time()
        fps = 0  # 当前FPS

        while self.running and not self.input_handler.is_exit_requested():
            try:
                # 如果切换到异步模式，退出同步循环
                if self.use_async:
                    self.logger.info("检测到模式切换，从同步切换到异步")
                    self.async_main_loop()
                    return

                # 开始总计时
                total_start = time.time()

                # 周期性检查资源使用情况 (每100帧检查一次，约3-4秒)
                memory_check_counter += 1
                if memory_check_counter >= 100:
                    memory_check_counter = 0
                    self.check_resources()

                # 周期性更新功能状态（每300帧更新一次，约10秒）
                feature_update_counter += 1
                if feature_update_counter >= 300:
                    feature_update_counter = 0
                    self.update_feature_display_states()

                # 捕获帧
                ret, frame = self.camera.read()
                if not ret:
                    self.handle_camera_error()
                    continue

                # 确保我们有一个有效的帧
                if frame is None or frame.size == 0:
                    self.logger.warning("获取到的帧无效，跳过")
                    time.sleep(0.1)
                    continue

                # 翻转帧并初始化
                frame = cv2.flip(frame, 1)
                current_time = time.time()

                # 人体姿态检测
                detection_start = time.time()
                persons = []
                try:
                    persons = self.detector.detect_pose(frame)
                    # 发布帧处理事件
                    self.events.publish("frame_captured", {"frame": frame,
                                                           "timestamp": current_time})
                except Exception as e:
                    self.logger.error(f"检测错误: {e}")
                    if current_time - self.last_error_time > self.error_cooldown:
                        print(f"检测错误: {e}")
                        self.last_error_time = current_time
                detection_time = time.time() - detection_start

                # 更新系统状态 - 使用SimplifiedSystemManager
                state_change = self.system_manager.update_state(
                    len(persons) > 0,
                    current_time)
                if state_change:
                    self.logger.info(f"状态变化: {state_change}")
                    # 发布系统状态变化事件
                    self.events.publish("system_state_changed", state_change)
                # 处理检测到的人或使用历史数据
                if persons:

                    person = persons[0]
                    self.events.publish("person_detected", {
                        'person': person,
                        'confidence': person.get('confidence', 0.5),
                        'frame': frame,
                        'timestamp': time.time()
                    })

                    # 发布人体检测事件
                    self.events.publish("person_detected", persons[0])

                    # 处理新检测到的人
                    try:
                        frame_viz, room_viz = self.process_detected_person(
                            persons[0], frame, current_time)

                        # 发布帧处理完成事件
                        self.events.publish("frame_processed", {
                            'frame_viz': frame_viz,
                            'room_viz': room_viz
                        })
                    except Exception as e:
                        self.logger.error(f"处理人体检测错误: {e}")
                        frame_viz = frame.copy()
                        room_viz = self.visualizer.visualize_room()
                    recovery_counter = 0
                elif self.system_manager.is_recent_detection(
                        current_time) and self.last_valid_person:
                    # 短暂检测失败，使用历史数据
                    try:
                        frame_viz = self.display_manager.draw_occlusion_message(
                            frame)
                        frame_viz, room_viz = self.process_occlusion(frame_viz,
                                                                     current_time)
                        # 发布遮挡状态事件
                        self.events.publish("occlusion_detected", {
                            'frame_viz': frame_viz,
                            'room_viz': room_viz
                        })
                    except Exception as e:
                        self.logger.error(f"处理遮挡状态错误: {e}")
                        frame_viz = frame.copy()
                        room_viz = self.visualizer.visualize_room()
                    recovery_counter += 1
                    if recovery_counter > 60:  # 约2秒没有恢复
                        self.logger.warning("长时间遮挡，重置追踪状态")
                        self.last_valid_person = None
                        recovery_counter = 0
                else:
                    # 长时间无检测，显示空房间
                    frame_viz = frame.copy()
                    room_viz = self.visualizer.visualize_room()
                    if self.last_valid_person:
                        self.logger.info("检测超时，重置状态")
                        self.last_valid_person = None

                # 确保显示前有有效的图像
                if 'frame_viz' not in locals() or frame_viz is None or frame_viz.size == 0:
                    frame_viz = frame.copy()
                if 'room_viz' not in locals() or room_viz is None or room_viz.size == 0:
                    room_viz = self.visualizer.visualize_room()

                # 更新FPS计算
                fps_counter += 1
                if current_time - fps_timer >= 1.0:  # 每秒更新一次FPS
                    fps = fps_counter / (current_time - fps_timer)
                    fps_counter = 0
                    fps_timer = current_time

                total_time = time.time() - total_start

                # 绘制调试信息并显示
                try:
                    frame_viz = self.display_manager.draw_debug_info(
                        frame_viz,
                        fps,
                        self.system_manager.get_current_state(),
                        self.position_mapper.get_debug_info() if hasattr(
                            self.position_mapper, 'get_debug_info') else None
                    )
                    self.display_manager.display_frames(frame_viz, room_viz)
                    # 发布UI更新事件
                    self.events.publish("ui_updated", {
                        'fps': fps,
                        'state': self.system_manager.get_current_state()
                    })
                except Exception as e:
                    self.logger.error(f"显示帧错误: {e}")
                    # 尝试显示原始帧
                    try:
                        cv2.imshow('Camera View', frame)
                    except Exception:
                        pass  # 如果还是失败，就放弃这一帧的显示

                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # 如果有按键
                    # 发布按键事件
                    self.events.publish("key_pressed", key)

                # 处理用户输入
                if self.input_handler.process_input():
                    break

            except Exception as e:
                self.logger.error(f"主循环错误: {e}", exc_info=True)
                # 添加更详细的错误信息
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(0.1)  # 防止错误循环太快

    def process_occlusion(self, frame_viz, current_time):
        """处理遮挡状态

        Args:
            frame_viz: 当前帧的可视化版本
            current_time: 当前时间戳

        Returns:
            tuple: (frame_viz, room_viz) 更新后的视图
        """
        person = self.last_valid_person
        action = self.last_valid_action
        room_viz = self.display_manager.get_last_room_viz()

        # 尝试使用位置预测
        if hasattr(self.position_mapper, 'predict_position'):
            try:
                # 计算自上次检测以来的时间
                elapsed_time = current_time - self.system_manager.last_detection_time
                predicted_pos = self.position_mapper.predict_position(
                    elapsed_time)

                if predicted_pos:
                    # 使用预测位置更新可视化
                    smooth_x, smooth_y = predicted_pos
                    self.visualizer.add_trail_point(smooth_x, smooth_y)

                    # 使用上次的深度或默认值
                    depth = person.get('depth', 5) if person else 5

                    room_viz = self.visualizer.visualize_room(
                        predicted_pos, depth, "预测位置")
                    self.display_manager.last_room_viz = room_viz
            except Exception as e:
                self.logger.error(f"位置预测错误: {e}")

        return frame_viz, room_viz

    def handle_camera_error(self):
        """处理相机读取错误"""
        current_time = time.time()
        if current_time - self.last_error_time > self.error_cooldown:
            self.logger.warning("相机读取错误, 尝试重新连接...")
            print("相机读取错误, 重试...")
            self.last_error_time = current_time

        time.sleep(0.1)  # 短暂延迟，避免CPU占用过高

    def create_async_pipeline(self):
        """创建异步处理管道"""
        try:
            # 检查相机是否已初始化
            if self.camera is None or not self.camera.isOpened():
                self.logger.error("无法创建异步管道: 相机未初始化")
                print("错误: 相机未初始化，无法创建异步管道")
                return None

            print("开始创建异步管道组件...")
            # 创建异步管道
            pipeline = AsyncPipeline(
                self.camera,
                self.detector,
                self.action_recognizer,
                self.position_mapper,
                self.system_manager
            )

            print("配置异步管道...")
            # 配置异步管道
            pipeline.configure(
                self.frame_width,
                self.frame_height,
                'balanced'  # 默认使用平衡模式
            )

            self.logger.info("异步管道创建成功")
            print("异步管道创建成功")
            return pipeline
        except Exception as e:
            self.logger.error(f"创建异步管道失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            print(f"异步管道创建失败: {e}")
            return None

    def check_resources(self):
        """检查资源使用情况并应用自适应策略"""
        try:
            # 使用SimplifiedSystemManager检查资源
            resource_status = self.system_manager.check_resources()

            # 根据资源状态应用自适应策略
            if self.system_manager.should_apply_adaptation():
                adaptation_level = self.system_manager.get_adaptation_level()
                suggestions = self.system_manager.get_adaptation_suggestions()

                self.logger.info(f"应用资源自适应策略，级别: {adaptation_level}")

                # 如果资源紧张且在同步模式，可以建议切换到异步模式
                if adaptation_level >= 2 and not self.use_async:
                    self.logger.warning("资源严重不足，建议切换到异步处理模式")
                    print("系统资源紧张，推荐使用异步处理模式（按'y'切换）")

                # 如果在异步模式，调整性能模式
                if self.use_async and self.async_pipeline:
                    if adaptation_level == 2:  # 资源紧张
                        if self.async_pipeline.performance_mode != 'high_speed':
                            self.async_pipeline.set_performance_mode(
                                'high_speed')
                            self.logger.warning(
                                "资源紧张，异步管道切换到高速模式")
                    elif adaptation_level == 1:  # 资源警告
                        if self.async_pipeline.performance_mode == 'high_quality':
                            self.async_pipeline.set_performance_mode('balanced')
                            self.logger.warning(
                                "资源警告，异步管道切换到平衡模式")

                # 应用建议的策略
                if suggestions['reduce_resolution']:
                    # 降低处理分辨率
                    if hasattr(self.detector, 'downscale_factor'):
                        current = self.detector.downscale_factor
                        # 根据级别决定降低的幅度
                        factor_reduction = 0.1 if adaptation_level == 1 else 0.2
                        new_factor = max(0.4, current * (1 - factor_reduction))
                        if new_factor < current:
                            self.detector.downscale_factor = new_factor
                            self.logger.info(
                                f"降低处理分辨率: {current:.2f} -> {new_factor:.2f}")

                # 禁用特定功能
                if 'mediapipe' in suggestions['disable_features'] and hasattr(
                        self.detector, 'using_mediapipe'):
                    if self.detector.using_mediapipe:
                        self.detector.using_mediapipe = False
                        self.display_manager.update_feature_state('mediapipe',
                                                                  False)
                        self.logger.warning("临时禁用MediaPipe功能")

                if 'ml_model' in suggestions['disable_features'] and hasattr(
                        self.action_recognizer, 'ml_recognizer'):
                    if self.action_recognizer.ml_recognizer is not None:
                        self.action_recognizer.ml_recognizer = None
                        self.display_manager.update_feature_state('ml_model',
                                                                  False)
                        self.logger.warning("临时禁用ML模型")

                # 清理历史数据
                if suggestions['clear_history']:
                    if hasattr(self.action_recognizer, 'keypoints_history'):
                        self.action_recognizer.keypoints_history.clear()
                    if hasattr(self.action_recognizer, 'position_history'):
                        self.action_recognizer.position_history.clear()
                    if hasattr(self.visualizer, 'trail_points'):
                        self.visualizer.trail_points.clear()
                    self.logger.warning("已清理历史数据缓存")

                # 强制垃圾回收
                if suggestions['force_gc']:
                    import gc
                    gc.collect()
                    self.logger.info("执行垃圾回收")

        except Exception as e:
            self.logger.error(f"资源检查错误: {e}")

    # 在TrackerApp类中添加这个方法
    def register_all_caches(self):
        """注册系统中的所有缓存实例"""
        cache_monitor = get_monitor()

        # 主要组件缓存
        if hasattr(self.detector, 'detection_cache'):
            cache_monitor.register_cache('detector',
                                         self.detector.detection_cache)

        if hasattr(self.action_recognizer, 'action_cache'):
            cache_monitor.register_cache('action_recognizer',
                                         self.action_recognizer.action_cache)

        if hasattr(self.position_mapper, 'position_cache'):
            cache_monitor.register_cache('position_mapper',
                                         self.position_mapper.position_cache)

        # 异步管道组件缓存
        if self.async_pipeline:
            # 检测处理器缓存
            if hasattr(self.async_pipeline.detection_processor,
                       'detection_cache'):
                cache_monitor.register_cache('async_detection',
                                             self.async_pipeline.detection_processor.detection_cache)

        # 检查SimplifiedDetector可能的内部缓存
        if hasattr(self.detector, '_cache'):
            cache_monitor.register_cache('detector_internal',
                                         self.detector._cache)

        # 记录注册的缓存数量
        self.logger.info(
            f"已注册 {len(cache_monitor.monitored_caches)} 个缓存实例")

    def update_feature_display_states(self):
        """更新功能显示状态"""
        # 为获取组件功能状态，我们需要各组件提供get_feature_state接口
        # 由于我们还没有为接口添加这个方法，使用安全的判断方式

        # 检查MediaPipe状态 - 通过检测器接口
        if hasattr(self.detector, 'get_feature_state'):
            # 如果接口支持get_feature_state方法
            mediapipe_state = self.detector.get_feature_state('mediapipe')
            self.display_manager.update_feature_state('mediapipe',
                                                      mediapipe_state)
        elif hasattr(self.detector, 'using_mediapipe'):
            # 向后兼容：直接访问属性
            self.display_manager.update_feature_state('mediapipe',
                                                      self.detector.using_mediapipe)

        # 检查ML模型状态 - 通过动作识别器接口
        if hasattr(self.action_recognizer, 'get_feature_state'):
            ml_state = self.action_recognizer.get_feature_state('ml_model')
            self.display_manager.update_feature_state('ml_model', ml_state)
        elif hasattr(self.action_recognizer, 'ml_recognizer'):
            self.display_manager.update_feature_state('ml_model',
                                                      self.action_recognizer.ml_recognizer is not None)

        # 检查DTW状态 - 通过动作识别器接口
        if hasattr(self.action_recognizer, 'get_feature_state'):
            dtw_state = self.action_recognizer.get_feature_state('dtw')
            self.display_manager.update_feature_state('dtw', dtw_state)
        elif hasattr(self.action_recognizer, 'dtw_recognizer'):
            self.display_manager.update_feature_state('dtw',
                                                      self.action_recognizer.dtw_recognizer is not None)

        # 检查多线程状态 - 通过动作识别器接口
        if hasattr(self.action_recognizer, 'get_feature_state'):
            threading_state = self.action_recognizer.get_feature_state(
                'threading')
            self.display_manager.update_feature_state('threading',
                                                      threading_state)
        elif hasattr(self.action_recognizer, 'config'):
            self.display_manager.update_feature_state('threading',
                                                      self.action_recognizer.config.get(
                                                          'enable_threading',
                                                          False))

        # 更新异步管道状态 - 这是TrackerApp自身的状态
        self.display_manager.update_feature_state('async', self.use_async)

    def _on_feature_toggled(self, data):
        """处理功能开关切换事件"""
        feature_name = data.get('feature_name')
        state = data.get('state')
        self.logger.info(
            f"功能 '{feature_name}' 已{'启用' if state else '禁用'}")
        # 在这里可以更新UI显示
        self.update_feature_display_states()

    def _on_calibration_completed(self, data):
        """处理校准完成事件"""
        calibration_height = data.get('height')
        self.logger.info(f"校准完成，参考高度: {calibration_height}")
        # 可以在这里执行校准后的操作

    def cleanup(self):
        """清理资源"""
        self.running = False
        self.logger.info("清理资源...")

        # 停止异步管道（如果有）
        if self.async_pipeline and self.async_pipeline.started:
            try:
                self.async_pipeline.stop()
                self.logger.info("异步管道已停止")
            except Exception as e:
                self.logger.error(f"停止异步管道时出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # 释放各组件资源
        if hasattr(self.detector, 'release_resources'):
            try:
                self.detector.release_resources()
                self.logger.info("检测器资源已释放")
            except Exception as e:
                self.logger.error(f"释放检测器资源时出错: {e}")

        if hasattr(self.action_recognizer, 'release_resources'):
            try:
                self.action_recognizer.release_resources()
                self.logger.info("动作识别器资源已释放")
            except Exception as e:
                self.logger.error(f"释放动作识别器资源时出错: {e}")

        if hasattr(self.position_mapper, 'release_resources'):
            try:
                self.position_mapper.release_resources()
                self.logger.info("位置映射器资源已释放")
            except Exception as e:
                self.logger.error(f"释放位置映射器资源时出错: {e}")

        # 停止事件日志记录器
        if hasattr(self, 'event_logger') and self.event_logger:
            self.event_logger.stop()
            self.logger.info("事件日志记录器已停止")


        # 释放相机
        if self.camera is not None:
            try:
                self.camera.release()
                self.logger.info("相机资源已释放")
            except Exception as e:
                self.logger.error(f"释放相机资源时出错: {e}")

        # 关闭所有窗口
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"关闭窗口时出错: {e}")

        # 强制垃圾回收
        import gc
        gc.collect()

        self.logger.info("程序已退出")
