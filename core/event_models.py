# -*- coding: utf-8 -*-
"""
事件数据模型模块 - 定义标准化的事件结构和类型
提供事件序列化和反序列化功能
"""

from enum import Enum, auto
from dataclasses import dataclass, field, asdict
import time
import json
import uuid
from typing import Any, Dict, Optional, Union, List, ClassVar, Type
import logging
from datetime import datetime

# 设置日志
logger = logging.getLogger("EventModels")


class EventType(Enum):
    """事件类型枚举 - 定义系统中所有标准事件类型"""

    # 系统事件
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    SYSTEM_ERROR = auto()
    SYSTEM_WARNING = auto()
    SYSTEM_INFO = auto()
    CONFIG_CHANGED = auto()
    RESOURCE_CRITICAL = auto()
    RESOURCE_WARNING = auto()

    # 组件生命周期事件
    COMPONENT_REGISTERED = auto()
    COMPONENT_INITIALIZED = auto()
    COMPONENT_STARTED = auto()
    COMPONENT_STOPPED = auto()
    COMPONENT_ERROR = auto()
    COMPONENT_STATE_CHANGED = auto()

    # 检测处理事件
    FRAME_CAPTURED = auto()
    PERSON_DETECTED = auto()
    PERSON_TRACKING_LOST = auto()
    ACTION_RECOGNIZED = auto()
    POSITION_MAPPED = auto()

    # UI相关事件
    UI_UPDATED = auto()
    UI_INPUT_RECEIVED = auto()
    UI_RENDER_COMPLETED = auto()
    UI_ERROR = auto()

    # 异步管道事件
    PIPELINE_STARTED = auto()
    PIPELINE_STOPPED = auto()
    PIPELINE_ERROR = auto()
    PIPELINE_STATE_CHANGED = auto()
    QUEUE_OVERFLOW = auto()

    # 特性切换事件
    FEATURE_ENABLED = auto()
    FEATURE_DISABLED = auto()
    FEATURE_ERROR = auto()

    # 性能和监控事件
    PERFORMANCE_THRESHOLD_EXCEEDED = auto()
    MEMORY_WARNING = auto()
    CPU_WARNING = auto()
    FPS_DROPPED = auto()
    LATENCY_WARNING = auto()

    # 用户交互事件
    USER_COMMAND_RECEIVED = auto()
    KEY_PRESSED = auto()
    CALIBRATION_STARTED = auto()
    CALIBRATION_COMPLETED = auto()

    # 自定义事件（用于扩展）
    CUSTOM_EVENT = auto()


class EventPriority(Enum):
    """事件优先级枚举"""
    LOW = 0  # 低优先级 - 后台任务、统计收集等
    NORMAL = 50  # 正常优先级 - 大多数事件
    HIGH = 100  # 高优先级 - 用户交互、关键状态变更
    CRITICAL = 200  # 关键优先级 - 错误、警告、系统稳定性相关


@dataclass
class EventMetadata:
    """事件元数据 - 包含事件的上下文信息"""

    # 事件源
    source_id: str = "system"  # 产生事件的组件ID

    # 事件标识和时间
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 唯一事件ID
    publication_time: float = field(default_factory=time.time)  # 发布时间戳
    delivery_time: Optional[float] = None  # 送达时间戳

    # 事件处理属性
    priority: EventPriority = EventPriority.NORMAL  # 事件优先级
    async_processing: bool = True  # 是否使用异步处理
    ttl: Optional[float] = None  # 生存时间（秒），None表示无限

    # 事件关联
    correlation_id: Optional[str] = None  # 相关联的事件ID
    causation_id: Optional[str] = None  # 因果关系ID
    session_id: Optional[str] = None  # 会话ID

    # 附加信息
    tags: List[str] = field(default_factory=list)  # 事件标签
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据

    def get_age(self) -> float:
        """获取事件年龄（秒）"""
        return time.time() - self.publication_time

    def is_expired(self) -> bool:
        """检查事件是否已过期"""
        if self.ttl is None:
            return False
        return self.get_age() > self.ttl

    def get_latency(self) -> Optional[float]:
        """获取事件延迟（从发布到送达的时间）"""
        if self.delivery_time is None:
            return None
        return self.delivery_time - self.publication_time

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换枚举为字符串
        data['priority'] = self.priority.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventMetadata':
        """从字典创建实例"""
        # 复制数据以避免修改原始数据
        data_copy = data.copy()

        # 转换字符串为枚举
        if 'priority' in data_copy and isinstance(data_copy['priority'], str):
            try:
                data_copy['priority'] = EventPriority[data_copy['priority']]
            except KeyError:
                data_copy['priority'] = EventPriority.NORMAL

        # 创建实例（忽略未知字段）
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data_copy.items() if
                         k in known_fields}

        return cls(**filtered_data)


@dataclass
class Event:
    """
    事件类 - 表示系统中的一个事件
    包含事件类型、数据和元数据
    """
    event_type: EventType  # 事件类型
    data: Any = None  # 事件数据
    metadata: EventMetadata = field(default_factory=EventMetadata)  # 事件元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_type': self.event_type.name,
            'data': self.data,
            'metadata': self.metadata.to_dict()
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        try:
            # 使用自定义编码器处理复杂对象
            return json.dumps(self.to_dict(), cls=EventJSONEncoder)
        except Exception as e:
            logger.error(f"事件JSON序列化错误: {e}")
            # 退回到基本序列化，可能会丢失部分数据
            return json.dumps({
                'event_type': self.event_type.name,
                'metadata': self.metadata.to_dict()
            })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建实例"""
        try:
            # 转换事件类型
            event_type_str = data.get('event_type')
            event_type = EventType[
                event_type_str] if event_type_str else EventType.CUSTOM_EVENT

            # 转换元数据
            metadata_dict = data.get('metadata', {})
            metadata = EventMetadata.from_dict(
                metadata_dict) if metadata_dict else EventMetadata()

            # 提取事件数据
            event_data = data.get('data')

            return cls(
                event_type=event_type,
                data=event_data,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"从字典创建事件失败: {e}")
            # 创建一个基本事件作为退路
            return cls(
                event_type=EventType.CUSTOM_EVENT,
                metadata=EventMetadata(
                    source_id="error_recovery",
                    metadata={'error': str(e), 'original_data': str(data)[:200]}
                )
            )

    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """从JSON字符串创建实例"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"从JSON创建事件失败: {e}, JSON: {json_str[:100]}...")
            # 创建一个错误事件作为退路
            return cls(
                event_type=EventType.SYSTEM_ERROR,
                data={'error': str(e), 'json_fragment': json_str[:100]},
                metadata=EventMetadata(source_id="json_parser")
            )


class EventJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于序列化事件和其他复杂对象"""

    def default(self, obj):
        # 处理Enum
        if isinstance(obj, Enum):
            return obj.name

        # 处理dataclass实例
        if hasattr(obj, "__dataclass_fields__"):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return asdict(obj)

        # 处理datetime和time
        if isinstance(obj, datetime):
            return obj.isoformat()

        # 处理UUID
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # 处理不可序列化的对象
        try:
            return str(obj)
        except:
            return repr(obj)

        # 让父类处理其他类型
        return super().default(obj)
