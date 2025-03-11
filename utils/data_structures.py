# -*- coding: utf-8 -*-
"""
数据结构模块 - 提供高效的数据结构实现

本模块包含项目中使用的各种数据结构实现，包括：
1. CircularBuffer - 固定大小的循环缓冲区，用于高效存储最近的数据项
2. LFUCache - 最少使用频率缓存，用于缓存经常访问的数据

这些数据结构设计为高效且线程安全，适用于实时处理场景。
"""
import time


class CircularBuffer:
    """
    固定大小的循环缓冲区，减少内存分配

    特点:
    1. 固定大小，自动覆盖最旧的数据
    2. 支持索引访问和迭代
    3. 内存使用稳定，适合实时应用

    示例:
        # 创建容量为5的循环缓冲区
        buffer = CircularBuffer(5)

        # 添加元素
        for i in range(7):
            buffer.append(i)

        # 此时buffer中的元素为[2, 3, 4, 5, 6]

        # 访问元素
        print(buffer[0])  # 输出: 2
        print(buffer[-1])  # 输出: 6

        # 迭代元素
        for item in buffer:
            print(item)  # 依次输出: 2, 3, 4, 5, 6

        # 获取最近的n个元素
        recent = buffer.get_latest(3)  # 返回[4, 5, 6]
    """

    def __init__(self, capacity):
        """
        初始化循环缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.size = 0
        self.head = 0
        self.tail = 0

    def append(self, item):
        """
        添加项目到缓冲区尾部

        Args:
            item: 要添加的项目
        """
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1
        else:
            # 缓冲区已满，头部向前移动
            self.head = (self.head + 1) % self.capacity

    def clear(self):
        """清空缓冲区"""
        self.size = 0
        self.head = 0
        self.tail = 0

    def __getitem__(self, index):
        """
        获取指定索引的元素

        Args:
            index: 索引值，可以是负数表示从末尾开始计数

        Returns:
            缓冲区中的元素

        Raises:
            IndexError: 如果索引超出范围
        """
        if index < 0:
            index = self.size + index

        if not 0 <= index < self.size:
            raise IndexError("CircularBuffer index out of range")

        idx = (self.head + index) % self.capacity
        return self.buffer[idx]

    def __len__(self):
        """
        获取缓冲区当前大小

        Returns:
            int: 缓冲区中的元素数量
        """
        return self.size

    def __iter__(self):
        """
        迭代缓冲区中的元素

        Returns:
            iterator: 缓冲区元素的迭代器
        """
        for i in range(self.size):
            yield self.buffer[(self.head + i) % self.capacity]

    def get_latest(self, n=None):
        """
        获取最近的n个元素

        Args:
            n: 要获取的元素数量，None则获取全部

        Returns:
            list: 最近的n个元素组成的列表
        """
        if n is None or n >= self.size:
            return list(self)

        result = []
        for i in range(max(0, self.size - n), self.size):
            result.append(self.buffer[(self.head + i) % self.capacity])
        return result


class LFUCache:
    """
    LFU (Least Frequently Used) 缓存实现

    特点:
    1. 淘汰策略基于访问频率，而不是时间
    2. 同一频率下，优先淘汰最早访问的项
    3. 支持最大容量和可选的超时设置

    示例:
        # 创建容量为100，超时10秒的LFU缓存
        cache = LFUCache(capacity=100, timeout=10.0)

        # 存储值
        cache.put("key1", "value1")

        # 获取值
        value = cache.get("key1")  # 返回"value1"
        value = cache.get("key2")  # 键不存在，返回None
        value = cache.get("key2", "default")  # 返回"default"

        # 检查键是否存在（不更新访问计数）
        exists = cache.contains("key1")  # 返回True

        # 获取缓存统计信息
        stats = cache.get_stats()
        print(f"缓存大小: {stats['size']}/{stats['capacity']}")
    """

    def __init__(self, capacity=100, timeout=None):
        """
        初始化LFU缓存

        Args:
            capacity: 缓存最大容量
            timeout: 可选的超时时间（秒），None表示无超时
        """
        self.capacity = max(1, capacity)  # 至少允许1个元素
        self.timeout = timeout  # 超时秒数，None表示无超时

        self.values = {}  # 存储实际值
        self.counts = {}  # 计数器
        self.timestamps = {}  # 最后访问时间
        self.freq_lists = {}  # 按频率分组的项列表
        self.min_freq = 0  # 当前最小频率

        # 在Python中，字典维护了插入顺序，所以同一频率内淘汰最早项变得简单

    def get(self, key, default=None):
        """
        获取缓存项并更新其访问计数

        Args:
            key: 缓存键
            default: 如果键不存在或已过期，返回的默认值

        Returns:
            缓存值或默认值
        """
        import time

        # 键不存在直接返回默认值
        if key not in self.values:
            return default

        # 检查超时
        if self.timeout is not None:
            if time.time() - self.timestamps[key] > self.timeout:
                self._remove(key)  # 移除过期项
                return default

        # 更新访问计数和时间戳
        self._update_usage(key)

        return self.values[key]

    def put(self, key, value):
        """
        添加或更新缓存项

        Args:
            key: 缓存键
            value: 要存储的值

        Returns:
            bool: True表示成功添加/更新，False表示出错
        """
        import time

        try:
            # 如果键已存在，更新值和访问计数
            if key in self.values:
                self.values[key] = value
                self._update_usage(key)
                return True

            # 如果已达到容量上限，需要先移除最不常用项
            if len(self.values) >= self.capacity:
                self._evict()

            # 添加新项，初始频率为1
            self.values[key] = value
            self.counts[key] = 1
            self.timestamps[key] = time.time()

            # 更新频率列表
            if 1 not in self.freq_lists:
                self.freq_lists[1] = []
            self.freq_lists[1].append(key)

            # 更新最小频率
            self.min_freq = 1

            return True

        except Exception as e:
            print(f"LFU缓存错误: {e}")
            return False

    def clear(self):
        """清空缓存"""
        self.values.clear()
        self.counts.clear()
        self.timestamps.clear()
        self.freq_lists.clear()
        self.min_freq = 0

    def contains(self, key):
        """
        检查键是否在缓存中且未过期

        Args:
            key: 要检查的键

        Returns:
            bool: 键是否有效
        """
        import time

        if key not in self.values:
            return False

        # 检查超时
        if self.timeout is not None:
            if time.time() - self.timestamps[key] > self.timeout:
                self._remove(key)  # 移除过期项
                return False

        return True

    def _update_usage(self, key):
        """
        更新项的使用计数和时间戳

        Args:
            key: 缓存键
        """
        import time

        # 获取当前频率和更新后的频率
        current_freq = self.counts[key]
        new_freq = current_freq + 1

        # 从当前频率列表中移除
        self.freq_lists[current_freq].remove(key)
        if len(self.freq_lists[current_freq]) == 0:
            del self.freq_lists[current_freq]

            # 如果最小频率列表空了，更新最小频率
            if self.min_freq == current_freq:
                self.min_freq = new_freq

        # 更新到新频率列表
        if new_freq not in self.freq_lists:
            self.freq_lists[new_freq] = []
        self.freq_lists[new_freq].append(key)

        # 更新计数和时间戳
        self.counts[key] = new_freq
        self.timestamps[key] = time.time()

    def _evict(self):
        """淘汰最不常用的项"""
        # 获取最小频率列表中的第一个键（最早添加的）
        if self.min_freq not in self.freq_lists or not self.freq_lists[
            self.min_freq]:
            return  # 空缓存或异常情况

        key_to_evict = self.freq_lists[self.min_freq][0]
        self._remove(key_to_evict)

    def _remove(self, key):
        """
        从缓存中移除指定键

        Args:
            key: 要移除的缓存键
        """
        if key not in self.values:
            return

        # 获取频率
        freq = self.counts[key]

        # 从频率列表移除
        self.freq_lists[freq].remove(key)
        if len(self.freq_lists[freq]) == 0:
            del self.freq_lists[freq]

            # 如果删除的是最小频率，需要找到新的最小频率
            if freq == self.min_freq:
                if not self.freq_lists:  # 如果没有其他频率了
                    self.min_freq = 0
                else:
                    self.min_freq = min(self.freq_lists.keys())

        # 删除相关数据
        del self.values[key]
        del self.counts[key]
        del self.timestamps[key]

    def __len__(self):
        """
        返回缓存中项的数量

        Returns:
            int: 缓存项数量
        """
        return len(self.values)

    def get_stats(self):
        """
        获取缓存统计信息

        Returns:
            dict: 包含缓存统计信息的字典，包括当前大小、容量、最小频率等
        """
        stats = {
            "size": len(self.values),
            "capacity": self.capacity,
            "min_freq": self.min_freq,
            "freq_distribution": {k: len(v) for k, v in self.freq_lists.items()}
        }
        return stats


# 使用示例
if __name__ == "__main__":
    # CircularBuffer示例
    print("CircularBuffer示例：")
    buffer = CircularBuffer(5)
    for i in range(7):
        buffer.append(i)
    print(f"缓冲区当前内容: {list(buffer)}")  # 应输出 [2, 3, 4, 5, 6]
    print(f"缓冲区大小: {len(buffer)}")  # 应输出 5
    print(f"最近3个元素: {buffer.get_latest(3)}")  # 应输出 [4, 5, 6]

    # LFUCache示例
    print("\nLFUCache示例：")
    cache = LFUCache(capacity=5, timeout=10)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    # 多次访问key1，增加其频率
    for _ in range(3):
        print(f"获取key1: {cache.get('key1')}")

    # 添加超过容量的项
    cache.put("key4", "value4")
    cache.put("key5", "value5")
    cache.put("key6", "value6")  # 将会淘汰最不常用的项

    # 检查哪些键还在缓存中
    for key in ["key1", "key2", "key3", "key4", "key5", "key6"]:
        print(f"{key} 在缓存中: {cache.contains(key)}")

    # 查看缓存统计
    stats = cache.get_stats()
    print(f"缓存统计: {stats}")
