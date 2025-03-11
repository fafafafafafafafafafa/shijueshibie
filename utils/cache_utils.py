# -*- coding: utf-8 -*-
"""
缓存工具模块 - 提供高效缓存实现和辅助函数

本模块提供:
1. 标准化缓存创建函数
2. 强健的缓存键生成方法
3. 缓存性能监控和日志记录
4. 缓存统计持久化支持
"""
import hashlib
import json
import logging
import time
import os
import threading
from datetime import datetime
import pickle

# 获取日志记录器
from .logger_config import get_logger

logger = get_logger("cache_utils")

# 全局缓存注册表
_cache_registry = {}
_registry_lock = threading.RLock()


def create_standard_cache(name="default", capacity=25, timeout=0.5,
                          persistent=False, stats_interval=3600):
    """
    创建标准化的LFU缓存实例

    Args:
        name: 缓存名称，用于日志和统计
        capacity: 缓存容量
        timeout: 缓存项超时时间（秒）
        persistent: 是否启用缓存统计持久化
        stats_interval: 统计信息持久化间隔（秒）

    Returns:
        LFUCache: 配置好的缓存实例
    """
    from .data_structures import LFUCache

    # 检查是否已存在同名缓存
    with _registry_lock:
        if name in _cache_registry:
            logger.warning(f"重复创建缓存 '{name}'，返回现有实例")
            return _cache_registry[name]

    logger.info(f"创建标准缓存: '{name}', 容量: {capacity}, 超时: {timeout}秒")

    # 创建缓存实例
    cache = LFUCache(capacity=capacity, timeout=timeout)

    # 增强缓存实例
    cache.name = name
    cache.creation_time = time.time()
    cache.hit_count = 0
    cache.miss_count = 0
    cache.last_stats_time = time.time()
    cache.persistent = persistent
    cache.stats_interval = stats_interval
    cache.stats_file = os.path.join("logs", "cache_stats", f"{name}_stats.json")

    # 添加访问计数方法
    original_get = cache.get

    def enhanced_get(key, default=None):
        result = original_get(key, default)
        if result is default:
            cache.miss_count += 1
            # 仅记录每10次未命中
            if cache.miss_count % 10 == 0:
                logger.debug(
                    f"缓存'{name}'未命中: {key}，总命中率: {cache.get_hit_rate():.2f}")
        else:
            cache.hit_count += 1
            # 仅记录每50次命中
            if cache.hit_count % 50 == 0:
                logger.debug(f"缓存'{name}'命中率: {cache.get_hit_rate():.2f}")

        # 检查是否应该持久化统计
        if persistent and (
                time.time() - cache.last_stats_time) > stats_interval:
            persist_cache_stats(cache)
            cache.last_stats_time = time.time()

        return result

    cache.get = enhanced_get

    # 添加命中率计算方法
    def get_hit_rate():
        total = cache.hit_count + cache.miss_count
        if total == 0:
            return 0.0
        return cache.hit_count / total

    cache.get_hit_rate = get_hit_rate

    # 添加到注册表
    with _registry_lock:
        _cache_registry[name] = cache

    # 创建统计目录（如果需要）
    if persistent:
        stats_dir = os.path.dirname(cache.stats_file)
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir, exist_ok=True)

    return cache


def generate_cache_key(data, prefix=""):
    """
    为缓存生成稳定的键

    使用增强的哈希算法，支持更多数据类型和更稳定的结果。

    Args:
        data: 用于生成键的数据（支持基本类型、列表、字典、numpy数组等）
        prefix: 可选前缀，用于区分不同类型的数据

    Returns:
        str: 生成的缓存键
    """
    try:
        # 处理None
        if data is None:
            return f"{prefix}:none"

        # 处理基本类型
        if isinstance(data, (str, int, float, bool)):
            key_str = f"{prefix}:{type(data).__name__}:{data}"
            return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:32]

        # 处理numpy数组（如果有）
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                # 对于大型数组，使用形状、类型和内容的统计摘要
                array_info = f"shape={data.shape},dtype={data.dtype}"
                if data.size <= 1000:  # 对于小数组，包含所有数据
                    array_info += f",data={data.tobytes()}"
                else:  # 对于大数组，包含统计摘要
                    array_info += f",mean={np.mean(data):.4f},std={np.std(data):.4f}"
                    array_info += f",min={np.min(data)},max={np.max(data)}"
                    array_info += f",hash={hash(data.tobytes()[:1000])}"

                key_str = f"{prefix}:ndarray:{array_info}"
                return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:32]
        except (ImportError, NameError):
            pass  # numpy不可用，继续其他处理

        # 处理字典、列表等
        try:
            # 尝试使用高效的msgpack序列化（如果可用）
            try:
                import msgpack
                serialized = msgpack.packb(data, use_bin_type=True)
                key_str = f"{prefix}:msgpack:{hashlib.sha256(serialized).hexdigest()}"
                return key_str[:32]
            except (ImportError, Exception):
                # 回退到JSON序列化
                json_str = json.dumps(data, sort_keys=True, default=str)
                key_str = f"{prefix}:json:{hashlib.sha256(json_str.encode('utf-8')).hexdigest()}"
                return key_str[:32]
        except:
            # 最后的回退方案：使用对象的字符串表示和类型
            obj_str = f"{prefix}:{type(data).__name__}:{str(data)}"
            return hashlib.sha256(obj_str.encode('utf-8')).hexdigest()[:32]

    except Exception as e:
        logger.error(f"生成缓存键时出错: {e}")
        # 生成一个基于错误和时间的后备键
        fallback = f"{prefix}:error:{time.time()}"
        return hashlib.sha256(fallback.encode('utf-8')).hexdigest()[:32]


def persist_cache_stats(cache):
    """
    将缓存统计信息持久化到文件

    Args:
        cache: 要持久化统计信息的缓存实例
    """
    if not getattr(cache, 'persistent', False):
        return

    try:
        # 获取统计信息
        stats = {
            "name": cache.name,
            "timestamp": datetime.now().isoformat(),
            "hit_count": cache.hit_count,
            "miss_count": cache.miss_count,
            "hit_rate": cache.get_hit_rate(),
            "size": len(cache),
            "capacity": cache.capacity,
            "uptime": time.time() - cache.creation_time
        }

        # 添加缓存特定统计信息
        if hasattr(cache, 'get_stats'):
            cache_stats = cache.get_stats()
            stats.update(cache_stats)

        # 确保目录存在
        stats_dir = os.path.dirname(cache.stats_file)
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir, exist_ok=True)

        # 写入文件
        with open(cache.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.debug(f"已持久化缓存'{cache.name}'的统计信息")

    except Exception as e:
        logger.error(f"持久化缓存'{cache.name}'的统计信息时出错: {e}")


def load_cache_stats(cache_name):
    """
    加载缓存统计信息

    Args:
        cache_name: 缓存名称

    Returns:
        dict: 加载的统计信息，如果不存在则返回None
    """
    stats_file = os.path.join("logs", "cache_stats", f"{cache_name}_stats.json")

    if not os.path.exists(stats_file):
        return None

    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载缓存'{cache_name}'的统计信息时出错: {e}")
        return None


def get_all_registered_caches():
    """
    获取所有已注册的缓存实例

    Returns:
        dict: 缓存名称到缓存实例的映射
    """
    with _registry_lock:
        return dict(_cache_registry)


def clear_all_caches():
    """
    清除所有注册的缓存

    Returns:
        int: 已清除的缓存数量
    """
    count = 0
    with _registry_lock:
        for name, cache in _cache_registry.items():
            try:
                # 保存最终统计信息
                if getattr(cache, 'persistent', False):
                    persist_cache_stats(cache)

                # 清除缓存内容
                cache.clear()
                count += 1
                logger.info(f"已清除缓存: '{name}'")
            except Exception as e:
                logger.error(f"清除缓存'{name}'时出错: {e}")

    return count


def save_cache_to_disk(cache, filepath):
    """
    将缓存内容保存到磁盘

    Args:
        cache: 要保存的缓存实例
        filepath: 保存路径

    Returns:
        bool: 保存是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 序列化并保存缓存
        with open(filepath, 'wb') as f:
            pickle.dump(cache.values, f)

        logger.info(f"已保存缓存'{cache.name}'到: {filepath}")
        return True

    except Exception as e:
        logger.error(f"保存缓存到磁盘时出错: {e}")
        return False


def load_cache_from_disk(cache, filepath):
    """
    从磁盘加载缓存内容

    Args:
        cache: 目标缓存实例
        filepath: 加载路径

    Returns:
        bool: 加载是否成功
    """
    if not os.path.exists(filepath):
        logger.warning(f"缓存文件不存在: {filepath}")
        return False

    try:
        # 加载并反序列化缓存
        with open(filepath, 'rb') as f:
            loaded_values = pickle.load(f)

        # 更新缓存
        cache.clear()
        for key, value in loaded_values.items():
            cache.put(key, value)

        logger.info(f"已从{filepath}加载缓存'{cache.name}'")
        return True

    except Exception as e:
        logger.error(f"从磁盘加载缓存时出错: {e}")
        return False
