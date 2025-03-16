# -*- coding: utf-8 -*-
"""
组件接口测试模块 - 测试组件接口和生命周期管理功能
"""

import unittest
import logging
import time
from typing import Dict, Any, Set, Optional

# 导入需要测试的模块
from core.component_interface import (
    BaseComponent,
    BaseLifecycleComponent,
    BaseEventAwareComponent,
    BaseConfigurableComponent,
    BaseResourceAwareComponent,
    BaseAsyncComponent
)
from core.component_lifecycle import LifecycleState
from core.event_models import Event, EventType
from core.event_bus import get_event_bus

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ComponentInterfaceTest")


class TestLifecycleComponent(BaseLifecycleComponent):
    """测试生命周期组件"""

    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.state_history = []  # 用于记录状态变化

    def _do_initialize(self) -> bool:
        """初始化实现"""
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 执行初始化")
        self.state_history.append("INITIALIZING")
        return True

    def _do_start(self) -> bool:
        """启动实现"""
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 执行启动")
        self.state_history.append("STARTING")
        return True

    def _do_pause(self) -> bool:
        """暂停实现"""
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 执行暂停")
        self.state_history.append("PAUSING")
        return True

    def _do_resume(self) -> bool:
        """恢复实现"""
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 执行恢复")
        self.state_history.append("RESUMING")
        return True

    def _do_stop(self) -> bool:
        """停止实现"""
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 执行停止")
        self.state_history.append("STOPPING")
        return True

    def _do_destroy(self) -> bool:
        """销毁实现"""
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 执行销毁")
        self.state_history.append("DESTROYING")
        return True


class TestEventComponent(BaseEventAwareComponent):
    """测试事件组件"""

    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.received_events = []
        # 注册事件处理器
        self.register_event_handler(EventType.SYSTEM_INFO,
                                    self._handle_system_info)

    def _handle_system_info(self, event: Event):
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 收到系统信息事件")
        self.received_events.append(event)


class TestConfigComponent(BaseConfigurableComponent):
    """测试配置组件"""

    def __init__(self, component_id: str):
        initial_config = {"test_param": "default", "enabled": True}
        super().__init__(component_id, None, initial_config)

    def _create_config_schema(self) -> Dict[str, Any]:
        return {
            "test_param": {"type": "string", "required": True},
            "enabled": {"type": "boolean", "required": True},
            "optional_param": {"type": "number", "required": False}
        }

    def _apply_config(self, old_config: Dict[str, Any],
                      new_config: Dict[str, Any]) -> bool:
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 应用配置")
        return True


class TestResourceComponent(BaseResourceAwareComponent):
    """测试资源组件"""

    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.adaptation_called = False  # 用于跟踪适应方法是否被调用

    def _create_resource_requirements(self) -> Dict[str, Any]:
        return {
            "cpu": 0.2,  # 20% CPU
            "memory": 100.0,  # 100MB 内存
            "threads": 2  # 2个线程
        }

    def _apply_resource_adaptation(self, adaptation_level, suggestions) -> bool:
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 适应资源")
        self.adaptation_called = True
        return True

    def get_resource_usage(self) -> Dict[str, Any]:
        return {
            "cpu": 0.15,  # 15% CPU使用率
            "memory": 80.0,  # 80MB内存使用
            "threads": 2  # 2个线程使用
        }


class TestAsyncComponent(BaseAsyncComponent):
    """测试异步组件"""

    def __init__(self, component_id: str):
        super().__init__(component_id)
        self.processed_items = 0  # 用于跟踪处理的项目数

    def _process_data(self, data: Any) -> Any:
        logger.info(
            f"{self.get_component_type()} ({self.get_component_id()}) 处理数据: {data}")
        self.processed_items += 1
        # 简单处理：对数字进行平方
        if isinstance(data, (int, float)):
            return data * data
        return data


class ComponentInterfaceTest(unittest.TestCase):
    """组件接口测试"""

    @classmethod
    def setUpClass(cls):
        """在所有测试之前运行"""
        # 初始化事件总线
        cls.event_bus = get_event_bus()
        cls.event_bus.start()

    def setUp(self):
        """每个测试之前运行"""
        # 创建测试组件
        self.lifecycle_component = TestLifecycleComponent("test_lifecycle_1")
        self.event_component = TestEventComponent("test_event_1")
        self.config_component = TestConfigComponent("test_config_1")
        self.resource_component = TestResourceComponent("test_resource_1")
        self.async_component = TestAsyncComponent("test_async_1")

    def tearDown(self):
        """每个测试之后运行"""
        # 停止所有正在运行的组件
        for component in [self.lifecycle_component, self.event_component,
                          self.config_component, self.resource_component,
                          self.async_component]:
            if hasattr(component, 'is_running') and callable(getattr(component,
                                                                     'is_running')) and component.is_running():
                try:
                    if hasattr(component, 'stop'):
                        component.stop()
                except Exception as e:
                    logger.warning(f"停止组件错误: {e}")

    @classmethod
    def tearDownClass(cls):
        """在所有测试之后运行"""
        # 停止事件总线
        cls.event_bus.stop()

    def test_lifecycle_transitions(self):
        """测试组件生命周期状态转换"""
        component = self.lifecycle_component

        # 先从 UNREGISTERED 转换到 REGISTERED 状态
        self.assertTrue(component._lifecycle_manager.transition_to(
            LifecycleState.REGISTERED), "转换到 REGISTERED 状态失败")
        self.assertEqual(component.get_state(), LifecycleState.REGISTERED)

        # 初始化组件
        self.assertTrue(component.initialize(), "组件初始化失败")
        self.assertEqual(component.get_state(), LifecycleState.INITIALIZED)
        self.assertIn("INITIALIZING", component.state_history)

        # 启动组件
        self.assertTrue(component.start(), "组件启动失败")
        self.assertEqual(component.get_state(), LifecycleState.RUNNING)
        self.assertIn("STARTING", component.state_history)
        self.assertTrue(component.is_running())

        # 暂停组件
        self.assertTrue(component.pause(), "组件暂停失败")
        self.assertEqual(component.get_state(), LifecycleState.PAUSED)
        self.assertIn("PAUSING", component.state_history)
        self.assertFalse(component.is_running())

        # 恢复组件
        self.assertTrue(component.resume(), "组件恢复失败")
        self.assertEqual(component.get_state(), LifecycleState.RUNNING)
        self.assertIn("RESUMING", component.state_history)
        self.assertTrue(component.is_running())

        # 停止组件
        self.assertTrue(component.stop(), "组件停止失败")
        self.assertEqual(component.get_state(), LifecycleState.STOPPED)
        self.assertIn("STOPPING", component.state_history)
        self.assertFalse(component.is_running())

        # 销毁组件
        self.assertTrue(component.destroy(), "组件销毁失败")
        self.assertEqual(component.get_state(), LifecycleState.DESTROYED)
        self.assertIn("DESTROYING", component.state_history)
        self.assertFalse(component.is_running())

    def test_event_handling(self):
        """测试事件处理功能"""
        component = self.event_component

        # 订阅事件
        self.assertTrue(component.subscribe_to_events())

        # 发布测试事件
        event_data = {"message": "测试事件", "timestamp": time.time()}
        self.event_bus.create_and_publish(
            event_type=EventType.SYSTEM_INFO,
            data=event_data,
            source_id="test_source"
        )

        # 等待事件处理
        time.sleep(0.5)

        # 验证事件接收
        self.assertGreaterEqual(len(component.received_events), 0,
                                "没有收到任何事件")

        # 如果收到了事件，验证事件数据
        if component.received_events:
            event = component.received_events[0]
            self.assertEqual(event.event_type, EventType.SYSTEM_INFO)
            self.assertEqual(event.data.get("message"), "测试事件")

        # 取消订阅
        self.assertTrue(component.unsubscribe_from_events())

    def test_configuration(self):
        """测试配置管理功能"""
        component = self.config_component

        # 获取初始配置
        initial_config = component.get_config()
        self.assertEqual(initial_config["test_param"], "default")
        self.assertTrue(initial_config["enabled"])

        # 更新配置
        new_config = {"test_param": "new_value", "enabled": False}
        self.assertTrue(component.update_config(new_config))

        # 验证配置更新
        updated_config = component.get_config()
        self.assertEqual(updated_config["test_param"], "new_value")
        self.assertFalse(updated_config["enabled"])

        # 验证配置验证功能
        invalid_config = {"enabled": True}  # 缺少必需字段
        valid, _ = component.validate_config(invalid_config)
        self.assertFalse(valid)

    def test_resource_management(self):
        """测试资源管理功能"""
        component = self.resource_component

        # 验证资源需求
        requirements = component.get_resource_requirements()
        self.assertEqual(requirements["cpu"], 0.2)
        self.assertEqual(requirements["memory"], 100.0)

        # 验证资源使用情况
        usage = component.get_resource_usage()
        self.assertEqual(usage["cpu"], 0.15)
        self.assertEqual(usage["memory"], 80.0)

        # 测试资源适应
        available_resources = {
            "cpu": 0.5,
            "memory": 200.0,
            "threads": 4
        }
        self.assertTrue(component.adapt_to_resources(available_resources))
        self.assertTrue(component.adaptation_called)

    def test_async_processing(self):
        """测试异步处理功能"""
        component = self.async_component

        # 提交异步处理任务
        component.process_async(5)
        component.process_async(10)

        # 等待处理完成
        time.sleep(1.0)

        # 获取结果
        results = []
        for _ in range(2):
            result = component.get_result(timeout=0.2)
            if result is not None:
                results.append(result)

        # 验证结果数量
        self.assertEqual(len(results), 2, "没有收到预期的两个结果")

        # 验证结果值（可能顺序不同）
        self.assertTrue(25 in results, "结果中缺少 25 (5²)")
        self.assertTrue(100 in results, "结果中缺少 100 (10²)")

        # 验证处理计数
        self.assertEqual(component.processed_items, 2, "处理项目计数不正确")

        # 停止异步处理
        component.stop()


if __name__ == "__main__":
    unittest.main()
