# -*- coding: utf-8 -*-
import cv2
import time
import logging
import numpy as np
import queue
import threading
import os

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
from ui.display_manager import DisplayManager as EnhancedDisplayManager
from ui.input_handler import InputHandler as EnhancedInputHandler
from config.config_integration import create_default_config
from core.component_lifecycle import LifecycleState

logger = setup_logger(__name__)


class TrackerApp:
    """主应用类，支持使用简化检测器的模块化架构，并具有异步处理能力"""

    def __init__(self, detector, action_recognizer, position_mapper,
                 visualizer, system_manager=None,config=None,event_logger=None,
                 display_manager=None, input_handler=None):
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
        # 首先初始化logger
        self.logger = logging.getLogger("TrackerApp")
        # 然后初始化配置系统
        self.config_system = self._init_config_system()

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
        self.use_async = self.config_system.get("system.async_mode", False)

        # 初始化相机
        self.camera = None
        self.frame_width = 0
        self.frame_height = 0
        self.init_camera()

        # 使用传入的UI组件或创建新的
        self.display_manager = display_manager or EnhancedDisplayManager(
            self.visualizer)
        self.input_handler = input_handler or EnhancedInputHandler(
            self.display_manager)

        # 确保组件正确注册
        self._register_lifecycle_components()

        # 在TrackerApp的__init__方法中添加:
        if hasattr(self.display_manager, 'initialize'):
            if not hasattr(self.display_manager,
                           '_initialized') or not self.display_manager._initialized:
                self.display_manager.initialize()
                if hasattr(self.display_manager, '_initialized'):
                    self.display_manager._initialized = True

        if hasattr(self.input_handler, 'initialize'):
            if not hasattr(self.input_handler,
                           '_initialized') or not self.input_handler._initialized:
                self.input_handler.initialize()
                if hasattr(self.input_handler, '_initialized'):
                    self.input_handler._initialized = True

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
        self._setup_input_handlers()
        self.register_all_caches()
        # 将组件绑定到配置系统
        self._bind_components_to_config()
        # 应用初始配置
        self._apply_initial_config()
        self.display_manager = EnhancedDisplayManager(self.visualizer)
        self.input_handler = EnhancedInputHandler(self.display_manager)

        # 注册输入处理函数
        self.input_handler.register_handler('r', self.recalibrate, '重新校准')
        self.input_handler.register_handler('h', self.show_help, '显示帮助')
        # 注册异步模式切换处理
        self.input_handler.register_handler('y', self.toggle_async_mode,
                                            '切换异步模式')
        # 添加配置相关的处理函数
        self.input_handler.register_handler('c', self.show_config_menu,
                                            '配置菜单')

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
        self._register_lifecycle_components()

    def _register_lifecycle_components(self):
        """注册所有生命周期组件"""
        # 注册显示管理器
        if hasattr(self.display_manager,
                   'get_state') and self.display_manager.get_state() == LifecycleState.UNREGISTERED:
            # 手动转换到 REGISTERED 状态
            if hasattr(self.display_manager, '_lifecycle_manager'):
                self.display_manager._lifecycle_manager.transition_to(
                    LifecycleState.REGISTERED)
                self.logger.info("display_manager 已注册")

        # 注册输入处理器
        if hasattr(self.input_handler,
                   'get_state') and self.input_handler.get_state() == LifecycleState.UNREGISTERED:
            # 手动转换到 REGISTERED 状态
            if hasattr(self.input_handler, '_lifecycle_manager'):
                self.input_handler._lifecycle_manager.transition_to(
                    LifecycleState.REGISTERED)
                self.logger.info("input_handler 已注册")

    # 添加配置系统初始化方法
    def _init_config_system(self):
        """初始化配置系统"""
        from config.config_system import get_config_system
        from config.config_integration import create_default_config

        default_config = create_default_config()

        config_dir = default_config

        # Define a proper directory path (for example)
        config_dir = os.path.join(os.path.expanduser('~'), '.pose_tracking')

        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            self.logger.info(f"创建配置目录: {config_dir}")

        config_system = get_config_system(
            config_dir=config_dir,
            default_config_file="pose_tracking_config.json",
            watch_interval=1.0,
            enable_file_watching=True,
            enable_schema_validation=True,
            enable_hot_reload=True
        )

        # 初始化默认配置模式
        config_system.init_default_schemas()

        # 尝试加载配置文件
        config_file_path = os.path.join(config_dir, "pose_tracking_config.json")
        if os.path.exists(config_file_path):
            try:
                config_system.load_config(config_file_path)
                self.logger.info(f"已加载配置文件: {config_file_path}")
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {e}")
                # 继续使用默认配置
        else:
            # 创建默认配置文件
            default_config = create_default_config()
            # 将默认配置直接保存到文件中
            with open(config_file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(default_config, f, indent=4, ensure_ascii=False)

            # 然后加载配置
            try:
                config_system.load_config(config_file_path)
                self.logger.info(f"已创建默认配置文件: {config_file_path}")
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {e}")
        # 监视配置目录
        config_system.watch_config_dir(pattern=r"\.json$")

        return config_system

    # 添加组件到配置系统的绑定方法
    def _bind_components_to_config(self):
        """将组件绑定到配置系统"""
        # 绑定检测器
        self.config_system.bind_component_config(
            self.detector,
            "detector",
            "detector"
        )

        # 绑定动作识别器
        self.config_system.bind_component_config(
            self.action_recognizer,
            "action_recognizer",
            "action_recognizer"
        )

        # 绑定位置映射器
        self.config_system.bind_component_config(
            self.position_mapper,
            "position_mapper",
            "position_mapper"
        )

        # 绑定系统管理器
        self.config_system.bind_component_config(
            self.system_manager,
            "system_manager",
            "system"
        )

        # 绑定可视化器
        self.config_system.bind_component_config(
            self.visualizer,
            "visualizer",
            "visualizer"
        )

        # 绑定显示管理器
        self.config_system.bind_component_config(
            self.display_manager,
            "display_manager",
            "ui"
        )

        self.logger.info("所有组件已绑定到配置系统")

    # 应用初始配置的方法
    def _apply_initial_config(self):
        """应用初始配置"""
        # 更新异步模式
        self.use_async = self.config_system.get("system.async_mode", False)

        # 更新特性状态
        self.update_feature_display_states()

        # 设置相机分辨率
        if self.camera is not None and self.camera.isOpened():
            width = self.config_system.get("ui.camera_width", 640)
            height = self.config_system.get("ui.camera_height", 480)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.logger.info(
                f"相机分辨率设置为: {self.frame_width}x{self.frame_height}")

        self.logger.info("已应用初始配置")

    # 添加配置菜单显示方法
    def show_config_menu(self):
        """显示配置菜单"""
        print("\n--- 配置菜单 ---")
        print("s - 保存当前配置")
        print("r - 重新加载配置")
        print("d - 重置为默认配置")
        print("q - 退出菜单")

        # 等待用户输入
        choice = input("请选择操作: ").lower()

        if choice == 's':
            self.config_system.save_config()
            print("配置已保存")
        elif choice == 'r':
            self.config_system.load_config()
            print("配置已重新加载")
        elif choice == 'd':
            self.config_system.reset_to_defaults()
            print("配置已重置为默认值")
        elif choice == 'q':
            print("退出配置菜单")
        else:
            print("无效选择")

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

        # 如果切换到异步模式，先检查异步管道的状态
        if new_state and self.async_pipeline:
            print(
                f"当前异步管道状态: pipeline={self.async_pipeline is not None}, started={getattr(self.async_pipeline, 'started', False)}")

        # 调用设置方法
        success = self.set_async_mode(new_state)

        if success:
            mode_name = "异步" if new_state else "同步"
            print(f"已成功切换到{mode_name}处理模式")

            # 直接更新 UI 状态
            self.display_manager.update_feature_state('async', new_state)

            # 尝试调用force_refresh
            try:
                if hasattr(self.display_manager, 'force_refresh'):
                    self.display_manager.force_refresh()
                    print("UI已强制刷新")
            except Exception as e:
                print(f"UI刷新时出错: {e}")

            # 重要：防止状态自动被重置
            self.use_async = new_state  # 再次确认状态

            # 设置一个标志，防止自动重置
            self._async_mode_toggled = True

            # 如果是切换到异步模式，再次检查管道状态
            if new_state:
                if self.async_pipeline:
                    print(
                        f"切换后异步管道状态: started={getattr(self.async_pipeline, 'started', False)}")
                    # 检查管道组件
                    if hasattr(self.async_pipeline, 'video_capture'):
                        print(
                            f"视频捕获状态: {getattr(self.async_pipeline.video_capture, 'running', False)}")
                    if hasattr(self.async_pipeline, 'detection_processor'):
                        print(
                            f"检测处理器状态: {getattr(self.async_pipeline.detection_processor, 'running', False)}")
                else:
                    print("异步管道对象不存在")

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

                # 再次检查以确保管道已创建
                if self.async_pipeline is None:
                    self.logger.error("创建异步管道失败，管道对象为None")
                    print("创建异步管道失败，管道对象为None")
                    return False

                # 如果应用正在运行，启动异步管道
                if self.running:
                    print("启动异步管道...")
                    try:
                        # 检查管道是否已经启动
                        if hasattr(self.async_pipeline,
                                   'started') and self.async_pipeline.started:
                            print("异步管道已经在运行中")
                        else:
                            self.async_pipeline.start()
                            # 验证管道是否真的启动了
                            if hasattr(self.async_pipeline, 'started'):
                                print(
                                    f"异步管道启动状态: {self.async_pipeline.started}")
                            else:
                                print("警告: 异步管道没有'started'属性")
                    except Exception as start_error:
                        print(f"启动异步管道时出错: {start_error}")
                        import traceback
                        print(traceback.format_exc())
                        return False

                # 检查管道的组件是否就绪
                if hasattr(self.async_pipeline, 'video_capture'):
                    print(
                        f"视频捕获器状态: started={getattr(self.async_pipeline.video_capture, 'thread', None) is not None}")
                if hasattr(self.async_pipeline, 'detection_processor'):
                    print(
                        f"检测处理器状态: started={getattr(self.async_pipeline.detection_processor, 'thread', None) is not None}")

                self.use_async = True
                self.logger.info("已切换到异步处理模式")
                print("已切换到异步处理模式")

                # 明确更新UI状态
                self.display_manager.update_feature_state('async', True)

                # 确保标志一致性
                self._async_mode_toggled = True  # 添加一个标志，表示刚刚切换了模式

            else:
                # 切换到同步模式
                if self.async_pipeline and self.running:
                    print("停止异步管道...")
                    try:
                        self.async_pipeline.stop()
                        print("异步管道已停止")
                    except Exception as stop_error:
                        print(f"停止异步管道时出错: {stop_error}")
                        import traceback
                        print(traceback.format_exc())

                self.use_async = False
                self.logger.info("已切换到同步处理模式")
                print("已切换到同步处理模式")

                # 明确更新UI状态
                self.display_manager.update_feature_state('async', False)

            return True
        except Exception as e:
            self.logger.error(f"切换异步模式失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            print(f"切换异步模式失败: {e}")
            print(traceback.format_exc())
            return False

    def debug_async_pipeline(self):
        """诊断异步管道的状态"""
        if self.async_pipeline is None:
            print("异步管道不存在")
            return

        print(f"异步管道对象: {self.async_pipeline}")
        print(
            f"异步管道已启动: {getattr(self.async_pipeline, 'started', False)}")

        if hasattr(self.async_pipeline, 'video_capture'):
            print(f"视频捕获状态: {self.async_pipeline.video_capture.__dict__}")

        if hasattr(self.async_pipeline, 'detection_processor'):
            print(
                f"检测处理器状态: {self.async_pipeline.detection_processor.__dict__}")

        if hasattr(self.async_pipeline, 'action_processor'):
            print(
                f"动作处理器状态: {self.async_pipeline.action_processor.__dict__}")

        if hasattr(self.async_pipeline, 'position_processor'):
            print(
                f"位置处理器状态: {self.async_pipeline.position_processor.__dict__}")

        print(f"异步模式标志: {self.use_async}")
        print(
            f"UI显示状态: {self.display_manager.feature_states.get('async', 'unknown')}")

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
        # 添加必要信息 - 确保包含校准高度信息
        if hasattr(self.position_mapper, 'calibration_height'):
            person[
                'calibration_height'] = self.position_mapper.calibration_height
        else:
            # 如果没有校准过，使用默认值
            person['calibration_height'] = person.get('height',
                                                      170)  # 默认身高170cm

        person['detection_time'] = current_time
        # 识别动作
        try:
            action = self.action_recognizer.recognize_action(person)
            self.last_valid_action = action
        except Exception as e:
            self.logger.error(f"动作识别错误: {e}")
            action = self.last_valid_action
            # 映射位置
        try:
            # 确保视图尺寸获取正确
            room_width = getattr(self.visualizer, 'room_width', 800)
            room_height = getattr(self.visualizer, 'room_height', 600)

            # 映射到房间坐标
            room_x, room_y, depth = self.position_mapper.map_position_to_room(
                self.frame_width, self.frame_height,
                room_width, room_height,
                person
            )
            # 确保获取到有效坐标
            if not isinstance(room_x, (int, float)) or not isinstance(room_y, (
            int, float)):
                self.logger.warning(
                    f"获取到无效坐标: {room_x}, {room_y}，使用默认值")
                room_x = room_width // 2
                room_y = room_height // 2

            # 获取稳定的位置
            if hasattr(self.position_mapper, 'get_stable_position'):
                stable_x, stable_y = self.position_mapper.get_stable_position(
                    room_x, room_y, action)
            else:
                stable_x, stable_y = room_x, room_y

            # 平滑位置
            if hasattr(self.position_mapper, 'smooth_position'):
                smooth_x, smooth_y = self.position_mapper.smooth_position(
                    stable_x, stable_y)
            else:
                smooth_x, smooth_y = stable_x, stable_y

            # 安全添加轨迹点
            if hasattr(self.visualizer, 'add_trail_point'):
                # 确保坐标是整数且在有效范围内
                trail_x = max(0, min(int(smooth_x), room_width))
                trail_y = max(0, min(int(smooth_y), room_height))
                self.visualizer.add_trail_point(trail_x, trail_y)

            # 可视化
            frame_viz = self.visualizer.visualize_frame(frame, person,
                                                        action,
                                                        self.detector)
            room_viz = self.visualizer.visualize_room((smooth_x, smooth_y),
                                                      depth, action)

            # 保存最后有效的房间视图
            if hasattr(self.display_manager, 'last_room_viz'):
                self.display_manager.last_room_viz = room_viz

        except Exception as e:
            self.logger.error(f"位置映射错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # 使用简单的备用可视化
            frame_viz = self.visualizer.visualize_frame(frame, person, action,
                                                        self.detector)
            if hasattr(self.display_manager, 'get_last_room_viz'):
                room_viz = self.display_manager.get_last_room_viz()
            else:
                room_viz = self.visualizer.visualize_room()  # 空房间

        return frame_viz, room_viz

    def _initialize_components(self):
        """确保所有组件已初始化和注册"""
        # 初始化和注册 display_manager
        if hasattr(self.display_manager, 'initialize'):
            self.display_manager.initialize()

        if hasattr(self.display_manager,
                   'get_state') and self.display_manager.get_state() == LifecycleState.UNREGISTERED:
            if hasattr(self.display_manager, '_lifecycle_manager'):
                self.display_manager._lifecycle_manager.transition_to(
                    LifecycleState.REGISTERED)
                # 进一步初始化它
                self.display_manager._lifecycle_manager.transition_to(
                    LifecycleState.INITIALIZING)
                self.display_manager._lifecycle_manager.transition_to(
                    LifecycleState.INITIALIZED)
                self.logger.info("display_manager 已完成初始化")

        # 初始化和注册 input_handler
        if hasattr(self.input_handler, 'initialize'):
            self.input_handler.initialize()

        if hasattr(self.input_handler,
                   'get_state') and self.input_handler.get_state() == LifecycleState.UNREGISTERED:
            if hasattr(self.input_handler, '_lifecycle_manager'):
                self.input_handler._lifecycle_manager.transition_to(
                    LifecycleState.REGISTERED)
                # 进一步初始化它
                self.input_handler._lifecycle_manager.transition_to(
                    LifecycleState.INITIALIZING)
                self.input_handler._lifecycle_manager.transition_to(
                    LifecycleState.INITIALIZED)
                self.logger.info("input_handler 已完成初始化")

    def run(self):
        """运行追踪应用"""
        # 首先进行校准
        # 确保组件已初始化和注册
        # 启动UI组件
        self._initialize_components()
        self._start_ui_components()
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
            # 主运行循环
            while self.running:
                if self.use_async:
                    print("准备进入异步模式")
                    # 确保异步管道正确创建和启动
                    if self.async_pipeline is None:
                        print("异步管道不存在，尝试创建")
                        self.async_pipeline = self.create_async_pipeline()
                        if not self.async_pipeline:
                            print("创建异步管道失败，回退到同步模式")
                            self.use_async = False
                            continue

                    # 确保异步管道已启动
                    if not hasattr(self.async_pipeline,
                                   'started') or not self.async_pipeline.started:
                        print("异步管道未启动，尝试启动")
                        self.async_pipeline.start()

                    # 调试异步管道状态
                    self.debug_async_pipeline()

                    # 运行异步主循环
                    self.async_main_loop()

                    # 如果异步循环结束但应用还在运行
                    if self.running:
                        print("异步主循环退出")
                        # 如果切换到了同步模式，继续下一次循环
                        if not self.use_async:
                            print("检测到切换到同步模式")
                            continue
                        # 如果仍在异步模式但循环退出，可能是出错了
                        else:
                            print("异步模式异常退出，重试")
                            time.sleep(0.5)  # 短暂延迟避免CPU占用过高
                            continue
                else:
                    print("进入同步模式")
                    self.sync_main_loop()

                    # 如果同步循环结束但应用还在运行
                    if self.running:
                        print("同步主循环退出")
                        # 如果切换到了异步模式，继续下一次循环
                        if self.use_async:
                            print("检测到切换到异步模式")
                            continue
                        # 如果仍在同步模式但循环退出，可能是出错了
                        else:
                            print("同步模式异常退出，重试")
                            time.sleep(0.5)
                            continue

                # 如果走到这里，说明用户要退出程序
                self.running = False

        except KeyboardInterrupt:
            self.logger.info("用户终止程序")
            print("\n用户终止程序")
        except Exception as e:
            self.logger.error(f"程序运行时发生错误: {e}", exc_info=True)
            print(f"程序运行时发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def _start_ui_components(self):
        """启动UI组件"""
        try:
            # 启动 display_manager
            if hasattr(self.display_manager,
                       'get_state') and self.display_manager.get_state() == LifecycleState.INITIALIZED:
                self.logger.info("启动 display_manager")
                self.display_manager.start()

            # 启动 input_handler
            if hasattr(self.input_handler,
                       'get_state') and self.input_handler.get_state() == LifecycleState.INITIALIZED:
                self.logger.info("启动 input_handler")
                self.input_handler.start()
        except Exception as e:
            self.logger.error(f"启动UI组件时出错: {e}")

    def sync_main_loop(self):
        """同步处理主循环"""
        # 初始化计数器和计时器
        fps_counter = 0
        fps_timer = time.time()
        fps = 0
        memory_check_counter = 0
        feature_update_counter = 0
        recovery_counter = 0

        self.logger.info("开始同步处理主循环")
        print("开始同步处理主循环")

        while self.running and not self.input_handler.is_exit_requested():
            try:
                # 如果切换到异步模式，退出同步循环
                if self.use_async:
                    self.logger.info("检测到模式切换，从同步切换到异步")
                    return

                # 开始总计时
                total_start = time.time()

                # 周期性检查资源使用情况
                memory_check_counter += 1
                if memory_check_counter >= 100:
                    memory_check_counter = 0
                    self.check_resources()

                # 周期性更新功能状态
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
                    self.events.publish("frame_captured", {
                        "frame": frame,
                        "timestamp": current_time
                    })
                except Exception as e:
                    self.logger.error(f"检测错误: {e}")
                    if current_time - self.last_error_time > self.error_cooldown:
                        print(f"检测错误: {e}")
                        self.last_error_time = current_time

                # 更新系统状态
                state_change = self.system_manager.update_state(
                    len(persons) > 0,
                    current_time)
                if state_change:
                    self.logger.info(f"状态变化: {state_change}")
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
                current_time = time.time()
                if current_time - fps_timer >= 1.0:  # 每秒更新一次FPS
                    fps = fps_counter / (current_time - fps_timer)
                    fps_counter = 0
                    fps_timer = current_time

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
                    self.events.publish("ui_updated", {
                        'fps': fps,
                        'state': self.system_manager.get_current_state()
                    })
                except Exception as e:
                    self.logger.error(f"显示帧错误: {e}")
                    try:
                        cv2.imshow('Camera View', frame)
                    except Exception:
                        pass  # 如果还是失败，就放弃这一帧的显示

                # 处理用户输入
                if self.input_handler.process_input():
                    break

            except Exception as e:
                self.logger.error(f"同步主循环错误: {e}", exc_info=True)
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(0.1)  # 防止错误循环太快

        self.logger.info("同步主循环结束")
        print("同步主循环结束")

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
            # 初始化并配置异步管道
            pipeline.initialize()
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
        #self.display_manager.update_feature_state('async', self.use_async)

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

    def async_main_loop(self):
        """异步处理主循环"""
        self.logger.info("开始异步处理主循环")
        print("开始异步处理主循环")

        # 检查异步管道是否正确初始化
        if self.async_pipeline is None:
            self.logger.error("异步管道对象不存在")
            print("异步管道对象不存在")
            # 回退到同步模式
            self.use_async = False
            self.display_manager.update_feature_state('async', False)
            return

        # 确保异步管道已经启动
        if not hasattr(self.async_pipeline,
                       'started') or not self.async_pipeline.started:
            self.logger.error("异步管道未启动或不可用")
            print("异步管道未启动或不可用")
            # 回退到同步模式，但不递归调用
            self.use_async = False
            self.display_manager.update_feature_state('async', False)
            return

        # 诊断计数器
        frame_counter = 0
        last_report_time = time.time()
        frames_received = 0

        # 记录开始时间用于FPS计算
        fps_counter = 0
        fps_timer = time.time()
        fps = 0

        try:
            while self.running and not self.input_handler.is_exit_requested():
                # 从异步管道获取处理结果
                try:
                    # 使用正确的方法获取结果
                    result = self.async_pipeline.get_latest_result()
                    if result:
                        frames_received += 1

                        # 从原始数据生成可视化
                        frame = result.get('frame')
                        person = result.get('person')
                        action = result.get('action', 'Static')
                        position = result.get('position')
                        action = result.get('action', 'Static')

                        if frame is not None and person is not None:
                            # 处理原始数据生成可视化内容
                            frame_viz, room_viz = self.process_detected_person(
                                person, frame, time.time())

                        print(
                            f"接收到异步结果 #{frames_received}, 类型: {type(result)}")

                        if frame is not None:
                            # 生成可视化内容
                            if person is not None:
                                # 处理检测到的人
                                try:
                                    frame_viz, room_viz = self.process_detected_person(
                                        person, frame, time.time())
                                except Exception as e:
                                    self.logger.error(f"处理检测结果错误: {e}")
                                    frame_viz = frame.copy()
                                    room_viz = self.visualizer.visualize_room()
                            else:
                                # 无人检测到，使用空房间
                                frame_viz = frame.copy()
                                room_viz = self.visualizer.visualize_room()

                            # 更新FPS计算
                            fps_counter += 1
                            current_time = time.time()
                            if current_time - fps_timer >= 1.0:  # 每秒更新一次FPS
                                fps = fps_counter / (current_time - fps_timer)
                                print(f"异步模式FPS: {fps:.1f}")
                                fps_counter = 0
                                fps_timer = current_time

                            # 绘制调试信息
                            frame_viz = self.display_manager.draw_debug_info(
                                frame_viz,
                                fps,
                                self.system_manager.get_current_state(),
                                self.position_mapper.get_debug_info() if hasattr(
                                    self.position_mapper,
                                    'get_debug_info') else None
                            )

                            # 显示帧
                            self.display_manager.display_frames(frame_viz,
                                                                room_viz)

                except queue.Empty:
                    # 队列为空，继续等待
                    frame_counter += 1
                    pass
                except Exception as e:
                    self.logger.error(f"处理异步结果时出错: {e}")
                    print(f"处理异步结果时出错: {e}")
                    import traceback
                    print(traceback.format_exc())

                # 定期报告状态
                current_time = time.time()
                if current_time - last_report_time > 5.0:
                    stats = self.async_pipeline.get_pipeline_stats() if hasattr(
                        self.async_pipeline, 'get_pipeline_stats') else {}
                    print(
                        f"异步管道状态: 循环次数={frame_counter}, 接收帧数={frames_received}")
                    print(f"管道统计: {stats}")
                    last_report_time = current_time
                    frame_counter = 0

                # 处理用户输入
                if self.input_handler.process_input():
                    break

                # 如果切换回同步模式，退出异步循环
                if not self.use_async:
                    self.logger.info("检测到模式切换，从异步切换到同步")
                    return  # 只返回，不递归调用

                # 短暂休眠，减轻CPU负担
                time.sleep(0.001)

        except Exception as e:
            self.logger.error(f"异步主循环错误: {e}", exc_info=True)
            print(f"异步主循环错误: {e}")
            # 添加更详细的错误信息
            import traceback
            print(traceback.format_exc())

        self.logger.info("异步主循环结束")
        print("异步主循环结束")

    def cleanup(self):
        """清理资源"""
        self.running = False
        self.logger.info("清理资源...")

        #帮助函数安全停止组件

        def safe_lifecycle_stop(component, component_name):
            try:
                if not component:
                    self.logger.info(f"{component_name} 不存在，跳过停止")
                    return

                # 检查组件的生命周期状态
                if hasattr(component, 'get_state'):
                    state = component.get_state()
                    self.logger.info(f"{component_name} 当前状态: {state}")

                    # 只有处于有效状态时才尝试停止
                    if state in [LifecycleState.RUNNING, LifecycleState.PAUSED,
                                 LifecycleState.ERROR]:
                        self.logger.info(
                            f"正在停止 {component_name} (状态: {state})")
                        component.stop()
                    else:
                        self.logger.info(
                            f"跳过 {component_name} 的停止 (状态: {state} 不支持停止操作)")

                # 如果组件没有生命周期状态但有停止方法，尝试调用清理方法
                elif hasattr(component, 'cleanup'):
                    self.logger.info(f"调用 {component_name} 的 cleanup() 方法")
                    component.cleanup()
                # 最后尝试直接关闭
                elif hasattr(component, 'close'):
                    self.logger.info(f"调用 {component_name} 的 close() 方法")
                    component.close()

            except Exception as e:
                self.logger.warning(f"停止 {component_name} 时出错: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        # 安全停止 input_handler
        if hasattr(self, 'input_handler'):
            safe_lifecycle_stop(self.input_handler, "input_handler")

        # 安全停止 display_manager
        if hasattr(self, 'display_manager'):
            safe_lifecycle_stop(self.display_manager, "display_manager")

        # 停止异步管道(如果存在)
        if hasattr(self, 'async_pipeline') and self.async_pipeline and getattr(
                self.async_pipeline, 'started', False):
            try:
                self.async_pipeline.stop()
                self.logger.info("异步管道已停止")
            except Exception as e:
                self.logger.error(f"停止异步管道时出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # 释放组件资源
        for component_name, component in [
            ("detector", self.detector if hasattr(self, 'detector') else None),
            ("action_recognizer", self.action_recognizer if hasattr(self,
                                                                    'action_recognizer') else None),
            ("position_mapper",
             self.position_mapper if hasattr(self, 'position_mapper') else None)
        ]:
            if component and hasattr(component, 'release_resources'):
                try:
                    component.release_resources()
                    self.logger.info(f"{component_name} 资源已释放")
                except Exception as e:
                    self.logger.error(f"释放 {component_name} 资源时出错: {e}")

        # 停止事件日志记录器
        if hasattr(self, 'event_logger') and self.event_logger:
            try:
                self.event_logger.stop()
                self.logger.info("事件日志记录器已停止")
            except Exception as e:
                self.logger.error(f"停止事件日志记录器时出错: {e}")

        # 关闭配置系统
        if hasattr(self, 'config_system'):
            try:
                self.config_system.shutdown()
                self.logger.info("配置系统已关闭")
            except Exception as e:
                self.logger.error(f"关闭配置系统时出错: {e}")

        # 释放相机资源
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.release()
                self.logger.info("相机资源已释放")
            except Exception as e:
                self.logger.error(f"释放相机资源时出错: {e}")

        # 关闭所有窗口
        try:
            cv2.destroyAllWindows()
            self.logger.info("所有窗口已关闭")
        except Exception as e:
            self.logger.error(f"关闭窗口时出错: {e}")

        # 强制垃圾回收
        try:
            import gc
            gc.collect()
            self.logger.info("垃圾回收已执行")
        except Exception as e:
            self.logger.error(f"执行垃圾回收时出错: {e}")

        self.logger.info("程序清理完成")
