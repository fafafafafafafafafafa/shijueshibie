# utils/event_serializer.py

import json
import time
import uuid
import zlib
import base64
from datetime import datetime
from collections import defaultdict
import copy


class EventRelationship:
    """事件关系类型"""
    CAUSES = "causes"  # 因果关系
    DEPENDS_ON = "depends_on"  # 依赖关系
    FOLLOWS = "follows"  # 顺序关系
    GROUPS_WITH = "groups_with"  # 分组关系
    CANCELS = "cancels"  # 取消关系
    MODIFIES = "modifies"  # 修改关系
    CONFLICTS_WITH = "conflicts_with"  # 冲突关系
    RESPONDS_TO = "responds_to"  # 响应关系


def serialize_event(event_type, event_data, include_metadata=True,
                    parent_id=None, relationships=None):
    """
    将事件序列化为JSON兼容的字典格式，支持事件关系和依赖

    Args:
        event_type: 事件类型
        event_data: 事件数据
        include_metadata: 是否包含额外元数据
        parent_id: 父事件ID（用于表示层次关系）
        relationships: 与其他事件的关系字典 {关系类型: [事件ID列表]}

    Returns:
        dict: 序列化后的事件字典
    """
    # 确保事件数据是字典类型
    if event_data is None:
        event_data = {}
    elif not isinstance(event_data, dict):
        event_data = {'value': event_data}

    # 生成唯一事件ID
    event_id = str(uuid.uuid4())

    # 获取当前时间戳
    current_time = time.time()

    # 添加标准元数据
    serialized_event = {
        'event_id': event_id,
        'event_type': event_type,
        'timestamp': event_data.get('timestamp', current_time),
        'recorded_at': current_time,
        'data': event_data
    }

    # 添加事件关系信息
    if parent_id:
        serialized_event['parent_id'] = parent_id

    if relationships:
        serialized_event['relationships'] = relationships

    # 添加可选的扩展元数据
    if include_metadata:
        import platform
        import os

        try:
            # 系统信息
            serialized_event['_metadata'] = {
                'system': platform.system(),
                'version': platform.version(),
                'python': platform.python_version(),
                'process_id': os.getpid()
            }

            # 尝试获取调用者的信息
            import inspect
            stack = inspect.stack()
            if len(stack) > 2:  # 0是当前函数, 1是调用者, 2是调用者的调用者
                caller = stack[2]
                serialized_event['_metadata']['caller'] = {
                    'file': os.path.basename(caller.filename),
                    'function': caller.function,
                    'line': caller.lineno
                }
        except Exception:
            # 获取元数据失败不应影响主要功能
            pass

    return serialized_event


def deserialize_event(serialized_event):
    """
    将序列化的事件字典转换回事件格式

    Args:
        serialized_event: 序列化的事件字典

    Returns:
        tuple: (事件类型, 事件数据, 事件ID, 关系字典) - 后两项可能为None
    """
    event_type = serialized_event['event_type']
    event_data = serialized_event['data']

    # 确保timestamp字段存在
    if 'timestamp' not in event_data and 'timestamp' in serialized_event:
        event_data['timestamp'] = serialized_event['timestamp']

    # 提取事件ID和关系信息（如果存在）
    event_id = serialized_event.get('event_id')
    relationships = serialized_event.get('relationships')

    return event_type, event_data, event_id, relationships


def format_timestamp(timestamp):
    """将时间戳格式化为人类可读的时间字符串"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')


def compress_event(serialized_event):
    """
    压缩序列化后的事件以减少存储空间

    Args:
        serialized_event: 序列化后的事件字典

    Returns:
        dict: 压缩后的事件字典
    """
    # 创建压缩版本
    compressed_event = {
        'event_id': serialized_event.get('event_id'),
        'event_type': serialized_event['event_type'],
        'timestamp': serialized_event['timestamp'],
        'recorded_at': serialized_event.get('recorded_at', time.time())
    }

    # 压缩数据部分
    if 'data' in serialized_event:
        data_str = json.dumps(serialized_event['data'], ensure_ascii=False)
        compressed_data = zlib.compress(data_str.encode('utf-8'))
        compressed_event['data_compressed'] = base64.b64encode(
            compressed_data).decode('ascii')

    # 添加父事件ID（如果存在）
    if 'parent_id' in serialized_event:
        compressed_event['parent_id'] = serialized_event['parent_id']

    # 压缩关系信息（如果存在）
    if 'relationships' in serialized_event:
        rel_str = json.dumps(serialized_event['relationships'],
                             ensure_ascii=False)
        compressed_rel = zlib.compress(rel_str.encode('utf-8'))
        compressed_event['relationships_compressed'] = base64.b64encode(
            compressed_rel).decode('ascii')

    # 压缩元数据（如果存在）
    if '_metadata' in serialized_event:
        metadata_str = json.dumps(serialized_event['_metadata'],
                                  ensure_ascii=False)
        compressed_metadata = zlib.compress(metadata_str.encode('utf-8'))
        compressed_event['_metadata_compressed'] = base64.b64encode(
            compressed_metadata).decode('ascii')

    return compressed_event


def decompress_event(compressed_event):
    """
    解压缩事件数据

    Args:
        compressed_event: 压缩后的事件字典

    Returns:
        dict: 解压后的事件字典
    """
    event = {
        'event_id': compressed_event.get('event_id'),
        'event_type': compressed_event['event_type'],
        'timestamp': compressed_event['timestamp'],
        'recorded_at': compressed_event.get('recorded_at')
    }

    # 解压数据
    if 'data_compressed' in compressed_event:
        compressed_data = base64.b64decode(compressed_event['data_compressed'])
        data_str = zlib.decompress(compressed_data).decode('utf-8')
        event['data'] = json.loads(data_str)

    # 添加父事件ID（如果存在）
    if 'parent_id' in compressed_event:
        event['parent_id'] = compressed_event['parent_id']

    # 解压关系信息
    if 'relationships_compressed' in compressed_event:
        compressed_rel = base64.b64decode(
            compressed_event['relationships_compressed'])
        rel_str = zlib.decompress(compressed_rel).decode('utf-8')
        event['relationships'] = json.loads(rel_str)

    # 解压元数据
    if '_metadata_compressed' in compressed_event:
        compressed_metadata = base64.b64decode(
            compressed_event['_metadata_compressed'])
        metadata_str = zlib.decompress(compressed_metadata).decode('utf-8')
        event['_metadata'] = json.loads(metadata_str)

    return event


class EventGraph:
    """事件关系图 - 用于表示和分析事件之间的关系"""

    def __init__(self):
        """初始化事件关系图"""
        self.events = {}  # 事件ID -> 事件数据
        self.relationships = defaultdict(list)  # 事件ID -> [(关系类型, 目标事件ID)]
        self.reverse_relationships = defaultdict(
            list)  # 事件ID -> [(关系类型, 源事件ID)]
        self.event_types = defaultdict(list)  # 事件类型 -> 事件ID列表

    def add_event(self, event_data):
        """
        添加事件到关系图

        Args:
            event_data: 序列化的事件字典

        Returns:
            str: 事件ID
        """
        event_id = event_data.get('event_id')
        if not event_id:
            event_id = str(uuid.uuid4())
            event_data['event_id'] = event_id

        # 存储事件
        self.events[event_id] = copy.deepcopy(event_data)

        # 更新事件类型索引
        event_type = event_data['event_type']
        self.event_types[event_type].append(event_id)

        # 处理关系
        if 'relationships' in event_data:
            for rel_type, target_ids in event_data['relationships'].items():
                for target_id in target_ids:
                    self.add_relationship(event_id, rel_type, target_id)

        # 处理父子关系
        if 'parent_id' in event_data:
            parent_id = event_data['parent_id']
            self.add_relationship(parent_id, 'contains', event_id)

        return event_id

    def add_relationship(self, source_id, relationship_type, target_id):
        """
        添加事件关系

        Args:
            source_id: 源事件ID
            relationship_type: 关系类型
            target_id: 目标事件ID
        """
        # 添加正向关系
        self.relationships[source_id].append((relationship_type, target_id))

        # 添加反向关系
        reverse_type = self._get_reverse_relationship(relationship_type)
        if reverse_type:
            self.reverse_relationships[target_id].append(
                (reverse_type, source_id))

    def _get_reverse_relationship(self, relationship_type):
        """获取关系的反向类型"""
        reverse_map = {
            EventRelationship.CAUSES: EventRelationship.DEPENDS_ON,
            EventRelationship.DEPENDS_ON: EventRelationship.CAUSES,
            EventRelationship.FOLLOWS: "precedes",
            EventRelationship.CANCELS: "canceled_by",
            EventRelationship.MODIFIES: "modified_by",
            EventRelationship.CONFLICTS_WITH: EventRelationship.CONFLICTS_WITH,
            EventRelationship.RESPONDS_TO: "responded_by",
            "contains": "contained_by"
        }
        return reverse_map.get(relationship_type)

    def get_related_events(self, event_id, relationship_type=None,
                           direction="outgoing"):
        """
        获取与指定事件相关的事件

        Args:
            event_id: 事件ID
            relationship_type: 关系类型，None表示所有类型
            direction: 关系方向 ("outgoing", "incoming", "both")

        Returns:
            list: 相关事件ID列表
        """
        related = []

        # 获取正向关系
        if direction in ["outgoing", "both"]:
            for rel_type, target_id in self.relationships[event_id]:
                if relationship_type is None or rel_type == relationship_type:
                    related.append(target_id)

        # 获取反向关系
        if direction in ["incoming", "both"]:
            for rel_type, source_id in self.reverse_relationships[event_id]:
                if relationship_type is None or rel_type == relationship_type:
                    related.append(source_id)

        return related

    def find_path(self, start_id, end_id, max_depth=10):
        """
        查找两个事件之间的关系路径

        Args:
            start_id: 起始事件ID
            end_id: 目标事件ID
            max_depth: 最大搜索深度

        Returns:
            list: 路径上的事件ID和关系列表，如 [(id1, rel1), (id2, rel2), ...]
        """
        if start_id == end_id:
            return [(start_id, None)]

        # 使用广度优先搜索
        visited = set([start_id])
        queue = [[(start_id, None)]]

        while queue and len(queue[0]) <= max_depth:
            path = queue.pop(0)
            current_id = path[-1][0]

            # 检查所有关联事件
            for rel_type, next_id in self.relationships[current_id]:
                if next_id == end_id:
                    # 找到目标
                    return path + [(next_id, rel_type)]

                if next_id not in visited:
                    visited.add(next_id)
                    queue.append(path + [(next_id, rel_type)])

        return None  # 未找到路径

    def get_causality_chain(self, event_id, max_depth=10):
        """
        获取事件的因果链

        Args:
            event_id: 事件ID
            max_depth: 最大链深度

        Returns:
            dict: 包含原因链和结果链的字典
        """
        # 获取原因链 (事件依赖的事件)
        causes_chain = []
        current_id = event_id
        depth = 0

        while depth < max_depth:
            causes = self.get_related_events(current_id,
                                             EventRelationship.DEPENDS_ON,
                                             "outgoing")
            if not causes:
                break

            # 选择第一个原因继续追踪
            current_id = causes[0]
            causes_chain.append(current_id)
            depth += 1

        # 获取结果链 (依赖于事件的事件)
        effects_chain = []
        current_id = event_id
        depth = 0

        while depth < max_depth:
            effects = self.get_related_events(current_id,
                                              EventRelationship.CAUSES,
                                              "outgoing")
            if not effects:
                break

            # 选择第一个结果继续追踪
            current_id = effects[0]
            effects_chain.append(current_id)
            depth += 1

        return {
            "causes": causes_chain,
            "effects": effects_chain
        }

    def get_event_dependencies(self, event_id):
        """
        获取事件的所有依赖

        Args:
            event_id: 事件ID

        Returns:
            dict: 依赖关系字典
        """
        # 直接依赖
        direct_depends = self.get_related_events(event_id,
                                                 EventRelationship.DEPENDS_ON,
                                                 "outgoing")

        # 间接依赖 (递归获取直接依赖的依赖)
        indirect_depends = []
        for dep_id in direct_depends:
            indirect = self.get_related_events(dep_id,
                                               EventRelationship.DEPENDS_ON,
                                               "outgoing")
            indirect_depends.extend(indirect)

        # 冲突事件
        conflicts = self.get_related_events(event_id,
                                            EventRelationship.CONFLICTS_WITH,
                                            "both")

        return {
            "direct_dependencies": direct_depends,
            "indirect_dependencies": indirect_depends,
            "conflicts": conflicts
        }

    def export_to_json(self, filepath):
        """
        将事件关系图导出为JSON文件

        Args:
            filepath: 输出文件路径

        Returns:
            bool: 是否成功导出
        """
        try:
            data = {
                "events": self.events,
                "relationships": {
                    source_id: {
                        "outgoing": [(rel_type, target_id) for
                                     rel_type, target_id in rels]
                    }
                    for source_id, rels in self.relationships.items()
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"导出关系图失败: {e}")
            return False

    @classmethod
    def import_from_json(cls, filepath):
        """
        从JSON文件导入事件关系图

        Args:
            filepath: 输入文件路径

        Returns:
            EventGraph: 事件关系图实例
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            graph = cls()

            # 添加事件
            for event_id, event_data in data["events"].items():
                graph.events[event_id] = event_data
                event_type = event_data["event_type"]
                graph.event_types[event_type].append(event_id)

            # 添加关系
            for source_id, rel_data in data["relationships"].items():
                for rel_type, target_id in rel_data.get("outgoing", []):
                    graph.add_relationship(source_id, rel_type, target_id)

            return graph
        except Exception as e:
            print(f"导入关系图失败: {e}")
            return None
