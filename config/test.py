#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器测试 - 验证ConfigManager类的核心功能

这个文件专注于测试ConfigManager类的基本功能，
包括配置的获取/设置、持久化和重置等核心特性。
"""

import os
import sys
import json
import shutil
import unittest
import logging

# 设置日志级别以查看详细信息
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入配置管理器
from config.config_manager import ConfigManager


class ConfigManagerTest(unittest.TestCase):
    """配置管理器测试类"""

    @classmethod
    def setUpClass(cls):
        """类级别的初始化，仅执行一次"""
        # 创建测试目录
        cls.test_dir = "cm_test_dir"
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        os.makedirs(cls.test_dir)

        # 初始化默认配置
        cls.default_config = {
            "app": {
                "name": "默认应用名称",
                "version": "1.0.0"
            },
            "settings": {
                "value": 100,
                "enabled": True
            }
        }

    @classmethod
    def tearDownClass(cls):
        """类级别的清理，仅执行一次"""
        # 删除测试目录
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def setUp(self):
        """每个测试方法前执行"""
        # 创建一个新的配置管理器实例
        self.config_manager = ConfigManager(
            config_dir=self.test_dir,
            default_config_file="test_config.json"
        )

        # 加载默认配置
        self.config_manager.load_default_config(self.default_config)

        # 记录日志
        logger.info("测试开始: %s", self._testMethodName)
        logger.info("初始默认配置: %s", self.default_config)

    def tearDown(self):
        """每个测试方法后执行"""
        # 记录日志
        logger.info("测试结束: %s", self._testMethodName)

        # 清理测试文件
        for file in os.listdir(self.test_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(self.test_dir, file))

    def test_01_get_set_basic(self):
        """测试基本的获取和设置功能"""
        # 测试获取默认值
        self.assertEqual(self.config_manager.get("app.name"), "默认应用名称")
        self.assertEqual(self.config_manager.get("settings.value"), 100)
        self.assertEqual(self.config_manager.get("settings.enabled"), True)

        # 测试设置值
        self.config_manager.set("app.name", "新应用名称")
        self.assertEqual(self.config_manager.get("app.name"), "新应用名称")

        self.config_manager.set("settings.value", 200)
        self.assertEqual(self.config_manager.get("settings.value"), 200)

        # 测试设置新的嵌套值
        self.config_manager.set("new.nested.key", "嵌套值")
        self.assertEqual(self.config_manager.get("new.nested.key"), "嵌套值")

        # 测试获取不存在的值
        self.assertIsNone(self.config_manager.get("non.existent"))
        self.assertEqual(self.config_manager.get("non.existent", "默认"),
                         "默认")

        # 获取整个配置
        config = self.config_manager.get_all_config()
        self.assertEqual(config["app"]["name"], "新应用名称")
        self.assertEqual(config["settings"]["value"], 200)
        self.assertEqual(config["new"]["nested"]["key"], "嵌套值")

    def test_02_save_load_file(self):
        """测试配置文件的保存和加载"""
        # 修改一些配置值
        self.config_manager.set("app.name", "文件测试")
        self.config_manager.set("settings.value", 999)

        # 保存到文件
        test_file = os.path.join(self.test_dir, "saved_config.json")
        success = self.config_manager.save_config_to_file(test_file)
        self.assertTrue(success)

        # 验证文件存在
        self.assertTrue(os.path.exists(test_file))

        # 读取文件内容
        with open(test_file, 'r', encoding='utf-8') as f:
            file_content = json.load(f)

        self.assertEqual(file_content["app"]["name"], "文件测试")
        self.assertEqual(file_content["settings"]["value"], 999)

        # 修改当前配置
        self.config_manager.set("app.name", "临时值")
        self.assertEqual(self.config_manager.get("app.name"), "临时值")

        # 从文件加载
        success = self.config_manager.load_config_from_file(test_file)
        self.assertTrue(success)

        # 验证配置已更新
        self.assertEqual(self.config_manager.get("app.name"), "文件测试")
        self.assertEqual(self.config_manager.get("settings.value"), 999)

    def test_03_reset_to_defaults(self):
        """测试重置到默认配置"""
        # 打印当前默认配置
        logger.info("测试前的默认配置: %s", self.config_manager._default_config)

        # 修改一些值
        self.config_manager.set("app.name", "已修改")
        self.config_manager.set("settings.value", 500)

        # 添加一个新配置项
        self.config_manager.set("new.key", "新值")

        # 验证修改已生效
        self.assertEqual(self.config_manager.get("app.name"), "已修改")
        self.assertEqual(self.config_manager.get("settings.value"), 500)
        self.assertEqual(self.config_manager.get("new.key"), "新值")

        # 重置到默认值
        self.config_manager.reset_to_defaults()

        # 验证重置后的值
        self.assertEqual(self.config_manager.get("app.name"), "默认应用名称")
        self.assertEqual(self.config_manager.get("settings.value"), 100)
        self.assertEqual(self.config_manager.get("settings.enabled"), True)

        # 验证新添加的配置项不再存在
        self.assertIsNone(self.config_manager.get("new.key"))

    def test_04_reset_after_load(self):
        """测试在加载文件后重置到默认配置"""
        # 修改配置并保存到文件
        self.config_manager.set("app.name", "文件中的名称")
        test_file = os.path.join(self.test_dir, "file_to_load.json")
        self.config_manager.save_config_to_file(test_file)

        # 初始状态记录
        original_default_name = self.config_manager._default_config["app"][
            "name"]
        logger.info("原始默认名称: %s", original_default_name)

        # 加载文件
        self.config_manager.load_config_from_file(test_file)
        logger.info("加载文件后的名称: %s", self.config_manager.get("app.name"))

        # 重置到默认值
        self.config_manager.reset_to_defaults()
        reset_name = self.config_manager.get("app.name")
        logger.info("重置后的名称: %s", reset_name)

        # 验证是否重置到正确的默认值
        self.assertEqual(reset_name, original_default_name)

    def test_05_nested_config_operations(self):
        """测试嵌套配置操作"""
        # 使用路径设置深层嵌套值
        self.config_manager.set("a.b.c.d.e", "深层值")

        # 验证能够获取
        self.assertEqual(self.config_manager.get("a.b.c.d.e"), "深层值")

        # 重置到默认值
        self.config_manager.reset_to_defaults()

        # 验证深层值不再存在
        self.assertIsNone(self.config_manager.get("a.b.c.d.e"))

    def test_06_notification(self):
        """测试变更通知"""
        # 记录通知
        notifications = []

        # 通知回调
        def on_change(key, old_value, new_value):
            notifications.append({
                "key": key,
                "old": old_value,
                "new": new_value
            })

        # 订阅变更
        self.config_manager.subscribe("test_subscriber", "app.name", on_change)

        # 修改配置
        self.config_manager.set("app.name", "新值通知测试")

        # 验证通知已发送
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["key"], "app.name")
        self.assertEqual(notifications[0]["old"], "默认应用名称")
        self.assertEqual(notifications[0]["new"], "新值通知测试")

        # 修改不相关的配置
        self.config_manager.set("settings.value", 777)

        # 验证没有新通知
        self.assertEqual(len(notifications), 1)

        # 取消订阅
        self.config_manager.unsubscribe("test_subscriber")

        # 再次修改app.name
        self.config_manager.set("app.name", "再次修改")

        # 验证没有新通知
        self.assertEqual(len(notifications), 1)


if __name__ == "__main__":
    unittest.main()
