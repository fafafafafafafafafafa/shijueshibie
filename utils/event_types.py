# utils/event_types.py
class EventTypes:
    """事件类型常量"""
    # 系统事件
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"

    # 检测事件
    FRAME_CAPTURED = "frame_captured"
    PERSON_DETECTED = "person_detected"
    DETECTION_FAILED = "detection_failed"

    # 动作事件
    ACTION_RECOGNIZED = "action_recognized"
    MOVEMENT_DETECTED = "movement_detected"

    # 位置事件
    POSITION_MAPPED = "position_mapped"
    POSITION_PREDICTION = "position_prediction"

    # UI事件
    KEY_PRESSED = "key_pressed"
    DISPLAY_UPDATED = "display_updated"
    FEATURE_TOGGLED = "feature_toggled"

    # 缓存事件
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"

    # 资源事件
    RESOURCE_WARNING = "resource_warning"
    RESOURCE_CRITICAL = "resource_critical"
