# utils/event_serializer.py

import json
import time
from datetime import datetime


def serialize_event(event_type, event_data):
    """
    将事件序列化为JSON兼容的字典格式

    Args:
        event_type: 事件类型
        event_data: 事件数据

    Returns:
        dict: 序列化后的事件字典
    """
    # 确保事件数据是字典类型
    if event_data is None:
        event_data = {}
    elif not isinstance(event_data, dict):
        event_data = {'value': event_data}

    # 添加标准元数据
    serialized_event = {
        'event_type': event_type,
        'timestamp': event_data.get('timestamp', time.time()),
        'recorded_at': time.time(),
        'data': event_data
    }

    return serialized_event


def deserialize_event(serialized_event):
    """
    将序列化的事件字典转换回事件格式

    Args:
        serialized_event: 序列化的事件字典

    Returns:
        tuple: (事件类型, 事件数据)
    """
    event_type = serialized_event['event_type']
    event_data = serialized_event['data']

    # 确保timestamp字段存在
    if 'timestamp' not in event_data and 'timestamp' in serialized_event:
        event_data['timestamp'] = serialized_event['timestamp']

    return event_type, event_data


def format_timestamp(timestamp):
    """将时间戳格式化为人类可读的时间字符串"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
