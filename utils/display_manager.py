# ui/display_manager.py (扩展)
from utils.event_types import EventTypes

class DisplayManager:
    # 已有代码...

    def setup_event_handlers(self, event_system):
        """设置事件处理器"""
        # 订阅相关事件
        event_system.subscribe(EventTypes.PERSON_DETECTED,
                               self.on_person_detected)
        event_system.subscribe(EventTypes.ACTION_RECOGNIZED,
                               self.on_action_recognized)
        event_system.subscribe(EventTypes.POSITION_MAPPED,
                               self.on_position_mapped)
        event_system.subscribe(EventTypes.FEATURE_TOGGLED,
                               self.on_feature_toggled)
        event_system.subscribe(EventTypes.SYSTEM_STATE_CHANGED,
                               self.on_system_state_changed)

    def on_person_detected(self, data):
        """人体检测事件处理"""
        # 更新UI显示
        if 'person' in data:
            # 可以在这里记录或处理，而不是等待主循环
            self._last_detected_person = data['person']

    def on_action_recognized(self, data):
        """动作识别事件处理"""
        if 'action' in data:
            self._last_action = data['action']

    def on_position_mapped(self, data):
        """位置映射事件处理"""
        if 'position' in data:
            self._last_position = data['position']

    def on_feature_toggled(self, data):
        """功能开关事件处理"""
        if 'feature_name' in data and 'state' in data:
            self.update_feature_state(data['feature_name'], data['state'])

    def on_system_state_changed(self, state):
        """系统状态变化事件处理"""
        self._system_state = state
