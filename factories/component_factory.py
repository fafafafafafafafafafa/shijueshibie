# factories/component_factory.py
from models.simplified_detector import SimplifiedPersonDetector

class ComponentFactory:
    """
    组件工厂类，负责创建各种组件
    """

    @staticmethod
    def create_detector(config, events=None):
        """
        创建检测器

        Args:
            config: 配置对象
            events: 事件系统实例，可选

        Returns:
            PersonDetectorInterface: 检测器实例
        """
        from models.simplified_detector import SimplifiedPersonDetector
        detector = SimplifiedPersonDetector(
            use_mediapipe=config.use_mediapipe,
            performance_mode=config.performance_mode,
            event_system=events
        )
        # 确保事件系统被设置
        if events and hasattr(detector, 'events'):
            detector.events = events
        return detector

    @staticmethod
    def create_action_recognizer(config, events=None):
        """
        创建动作识别器

        Args:
            config: 配置对象
            events: 事件系统实例，可选

        Returns:
            ActionRecognizerInterface: 动作识别器实例
        """
        from models.simplified_action_recognizer import SimplifiedActionRecognizer
        recognizer = SimplifiedActionRecognizer(
            config={
                'keypoint_confidence_threshold': config.keypoint_confidence_threshold,
                'motion_cooldown': config.motion_cooldown
            },
            event_system=events
        )
        # 确保事件系统被设置
        if events and hasattr(recognizer, 'events'):
            recognizer.events = events
        return recognizer

    @staticmethod
    def create_position_mapper(config, events=None, room_width=800, room_height=600):
        """
        创建位置映射器

        Args:
            config: 配置对象
            events: 事件系统实例，可选
            room_width: 房间宽度
            room_height: 房间高度

        Returns:
            PositionMapperInterface: 位置映射器实例
        """
        from models.simplified_position_mapper import SimplifiedPositionMapper
        mapper = SimplifiedPositionMapper(
            room_width=room_width,
            room_height=room_height,
            event_system=events
        )
        # 确保事件系统被设置
        if events and hasattr(mapper, 'events'):
            mapper.events = events
        return mapper

    @staticmethod
    def create_visualizer(room_width=800, room_height=600):
        """
        创建可视化器

        Args:
            room_width: 房间宽度
            room_height: 房间高度

        Returns:
            Visualizer: 可视化器实例
        """
        from ui.visualizer import Visualizer
        return Visualizer(
            room_width=room_width,
            room_height=room_height
        )

    @staticmethod
    def create_system_manager(config, events=None):
        """
        创建系统管理器

        Args:
            config: 配置对象
            events: 事件系统实例，可选

        Returns:
            SimplifiedSystemManager: 系统管理器实例
        """
        from core.SimplifiedSystemManager import SimplifiedSystemManager
        manager = SimplifiedSystemManager(
            log_interval=config.log_interval,
            memory_warning_threshold=config.memory_warning_threshold,
            memory_critical_threshold=config.memory_critical_threshold
        )
        # 确保事件系统被设置
        if events and hasattr(manager, 'events'):
            manager.events = events
        return manager

    @staticmethod
    def create_detector(config, events=None):
        from models.simplified_detector import SimplifiedPersonDetector

        detector = SimplifiedPersonDetector(
            use_mediapipe=config.get('use_mediapipe', False),
            performance_mode=config.get('performance_mode', 'balanced'),
            event_system=events,
            config=config
        )

        # 确保事件系统被设置
        if events and hasattr(detector, 'events'):
            detector.events = events

        # 添加类型检查
        from interface.detector_interface import PersonDetectorInterface
        if not isinstance(detector, PersonDetectorInterface):
            raise TypeError(f"创建的检测器类型错误: {type(detector)}")

        return detector
