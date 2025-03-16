# -
import logging
import threading
import time
from typing import Dict, Any, List, Optional
from enum import Enum
logger = logging.getLogger("ResourceManager")

class AdaptationLevel(Enum):
    """资源适应级别枚举"""
    NORMAL = 0      # 正常运行，无需适应
    LOW = 1         # 轻度资源压力，轻微适应
    MEDIUM = 2      # 中度资源压力，中等适应
    HIGH = 3        # 高度资源压力，强力适应
    CRITICAL = 4    # 临界资源压力，紧急适应


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    THREADS = "threads"

class ResourceManager:
    """
    资源管理器 - 管理系统资源分配和监控
    采用单例模式
    """

    # 单例实例
    _instance = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls):
        """
        获取资源管理器单例实例

        Returns:
            ResourceManager: 资源管理器实例
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = ResourceManager()
        return cls._instance

    def __init__(self):
        """初始化资源管理器"""
        # 防止直接实例化
        if ResourceManager._instance is not None:
            raise RuntimeError(
                "ResourceManager是单例类，请使用get_instance()获取实例")

        # 资源使用情况
        self._resource_usage = {
            ResourceType.CPU.value: 0.0,
            ResourceType.MEMORY.value: 0.0,
            ResourceType.DISK.value: 0.0,
            ResourceType.NETWORK.value: 0.0,
            ResourceType.GPU.value: 0.0,
            ResourceType.THREADS.value: 0
        }

        # 资源限制
        self._resource_limits = {
            ResourceType.CPU.value: 1.0,  # 100% CPU
            ResourceType.MEMORY.value: 1024.0,  # 1024MB 内存
            ResourceType.DISK.value: 10240.0,  # 10GB 磁盘
            ResourceType.NETWORK.value: 100.0,  # 100MB/s 网络
            ResourceType.GPU.value: 1.0,  # 100% GPU
            ResourceType.THREADS.value: 16  # 16个线程
        }

        # 资源分配
        self._resource_allocations = {}

        # 资源监控间隔（秒）
        self._monitor_interval = 5.0

        # 资源监控线程
        self._monitor_thread = None
        self._running = False

        # 当前适应级别
        self._adaptation_level = AdaptationLevel.NORMAL

        # 适应建议
        self._adaptation_suggestions = {}

        logger.info("资源管理器已初始化")

    def start_monitoring(self):
        """启动资源监控"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="ResourceMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("资源监控已启动")

    def stop_monitoring(self):
        """停止资源监控"""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
                self._monitor_thread = None
            logger.info("资源监控已停止")

    def _monitor_loop(self):
        """资源监控循环"""
        logger.info("资源监控线程已启动")

        while self._running:
            try:
                # 更新资源使用情况
                self._update_resource_usage()

                # 计算适应级别
                self._calculate_adaptation_level()

                # 生成适应建议
                self._generate_adaptation_suggestions()

                # 等待下一次监控
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                time.sleep(1.0)  # 出错时短暂等待

        logger.info("资源监控线程已停止")

    def _update_resource_usage(self):
        """更新资源使用情况"""
        # 实际实现应该从系统获取资源使用情况
        # 这里仅作为示例实现
        import random

        self._resource_usage[ResourceType.CPU.value] = random.uniform(0.1, 0.8)
        self._resource_usage[ResourceType.MEMORY.value] = random.uniform(100.0,
                                                                         800.0)
        self._resource_usage[ResourceType.DISK.value] = random.uniform(1000.0,
                                                                       5000.0)
        self._resource_usage[ResourceType.NETWORK.value] = random.uniform(1.0,
                                                                          50.0)
        self._resource_usage[ResourceType.GPU.value] = random.uniform(0.0, 0.5)
        self._resource_usage[ResourceType.THREADS.value] = random.randint(1, 10)

    def _calculate_adaptation_level(self):
        """计算适应级别"""
        # 计算资源使用率
        cpu_usage_ratio = self._resource_usage[ResourceType.CPU.value] / \
                          self._resource_limits[ResourceType.CPU.value]
        memory_usage_ratio = self._resource_usage[ResourceType.MEMORY.value] / \
                             self._resource_limits[ResourceType.MEMORY.value]

        # 确定适应级别
        if cpu_usage_ratio > 0.9 or memory_usage_ratio > 0.9:
            self._adaptation_level = AdaptationLevel.CRITICAL
        elif cpu_usage_ratio > 0.8 or memory_usage_ratio > 0.8:
            self._adaptation_level = AdaptationLevel.HIGH
        elif cpu_usage_ratio > 0.7 or memory_usage_ratio > 0.7:
            self._adaptation_level = AdaptationLevel.MEDIUM
        elif cpu_usage_ratio > 0.6 or memory_usage_ratio > 0.6:
            self._adaptation_level = AdaptationLevel.LOW
        else:
            self._adaptation_level = AdaptationLevel.NORMAL

    def _generate_adaptation_suggestions(self):
        """生成适应建议"""
        self._adaptation_suggestions = {}

        if self._adaptation_level == AdaptationLevel.NORMAL:
            return

        # 生成适应建议
        if self._resource_usage[ResourceType.CPU.value] / self._resource_limits[
            ResourceType.CPU.value] > 0.6:
            self._adaptation_suggestions[ResourceType.CPU.value] = {
                "reduce_usage": True,
                "target": self._resource_limits[ResourceType.CPU.value] * 0.5,
                "priority": self._adaptation_level.value
            }

        if self._resource_usage[ResourceType.MEMORY.value] / \
                self._resource_limits[ResourceType.MEMORY.value] > 0.6:
            self._adaptation_suggestions[ResourceType.MEMORY.value] = {
                "reduce_usage": True,
                "target": self._resource_limits[
                              ResourceType.MEMORY.value] * 0.5,
                "priority": self._adaptation_level.value
            }

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        获取资源使用情况

        Returns:
            Dict[str, Any]: 资源使用情况
        """
        with self._lock:
            return dict(self._resource_usage)

    def get_resource_limits(self) -> Dict[str, Any]:
        """
        获取资源限制

        Returns:
            Dict[str, Any]: 资源限制
        """
        with self._lock:
            return dict(self._resource_limits)

    def get_adaptation_level(self) -> AdaptationLevel:
        """
        获取当前适应级别

        Returns:
            AdaptationLevel: 适应级别
        """
        with self._lock:
            return self._adaptation_level

    def get_adaptation_suggestions(self) -> Dict[str, Any]:
        """
        获取适应建议

        Returns:
            Dict[str, Any]: 适应建议
        """
        with self._lock:
            return dict(self._adaptation_suggestions)

    def allocate_resources(self, component_id: str,
                           requirements: Dict[str, Any]) -> str:
        """
        分配资源

        Args:
            component_id: 组件ID
            requirements: 资源需求

        Returns:
            str: 分配ID
        """
        with self._lock:
            allocation_id = f"alloc_{component_id}_{int(time.time())}"

            # 检查是否有足够资源
            for resource_type, amount in requirements.items():
                if resource_type not in self._resource_limits:
                    continue

                available = self._resource_limits[resource_type] - \
                            self._resource_usage[resource_type]
                if amount > available:
                    logger.warning(
                        f"资源不足: {resource_type}, 需要: {amount}, 可用: {available}")
                    return None

            # 分配资源
            self._resource_allocations[allocation_id] = {
                "component_id": component_id,
                "requirements": requirements,
                "allocation_time": time.time()
            }

            # 更新资源使用情况
            for resource_type, amount in requirements.items():
                if resource_type in self._resource_usage:
                    self._resource_usage[resource_type] += amount

            logger.info(f"已分配资源: {allocation_id}, 组件: {component_id}")
            return allocation_id

    def release_resources(self, allocation_id: str) -> bool:
        """
        释放资源

        Args:
            allocation_id: 分配ID

        Returns:
            bool: 是否成功释放
        """
        with self._lock:
            if allocation_id not in self._resource_allocations:
                logger.warning(f"找不到资源分配: {allocation_id}")
                return False

            allocation = self._resource_allocations[allocation_id]

            # 更新资源使用情况
            for resource_type, amount in allocation["requirements"].items():
                if resource_type in self._resource_usage:
                    self._resource_usage[resource_type] -= amount

            # 删除分配记录
            del self._resource_allocations[allocation_id]

            logger.info(
                f"已释放资源: {allocation_id}, 组件: {allocation['component_id']}")
            return True

    def update_resource_limits(self, limits: Dict[str, Any]) -> bool:
        """
        更新资源限制

        Args:
            limits: 新限制

        Returns:
            bool: 是否成功更新
        """
        with self._lock:
            for resource_type, limit in limits.items():
                if resource_type in self._resource_limits:
                    self._resource_limits[resource_type] = limit

            logger.info(f"已更新资源限制")
            return True

if __name__ == "__main__":
    unittest.main()
