#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件监视器模块 - 监控配置文件变化并触发热重载

该模块提供了配置文件的监视功能，当配置文件发生变化时，通知系统重新加载配置。
它支持监视单个文件或整个目录的配置文件，并提供不同的监视策略。
"""

import os
import time
import logging
import threading
import hashlib
from typing import Dict, List, Set, Callable, Optional, Any, Tuple
from datetime import datetime
import queue
import re

logger = logging.getLogger(__name__)


class WatchMode:
    """监视模式枚举"""
    POLL = "poll"  # 轮询模式
    EVENT = "event"  # 事件模式（如果可用）
    HYBRID = "hybrid"  # 混合模式


class ConfigWatcher:
    """配置文件监视器"""

    def __init__(
            self,
            mode: str = WatchMode.POLL,
            interval: float = 1.0,
            auto_reload: bool = True
    ):
        """初始化配置监视器

        Args:
            mode: 监视模式
            interval: 检查间隔（秒）
            auto_reload: 是否自动重新加载配置
        """
        self.mode = mode
        self.interval = interval
        self.auto_reload = auto_reload

        # 监视的文件和目录
        self._watched_files: Dict[str, str] = {}  # path -> hash
        self._watched_dirs: Dict[
            str, Dict[str, str]] = {}  # dir -> {file -> hash}

        # 变更回调
        self._file_callbacks: Dict[str, List[Callable[[str], None]]] = {}
        self._dir_callbacks: Dict[str, List[Callable[[str], None]]] = {}
        self._global_callbacks: List[Callable[[str], None]] = []

        # 监视线程
        self._should_stop = threading.Event()
        self._watch_thread = None
        self._change_queue = queue.Queue()
        self._notification_thread = None

        # 状态
        self._running = False
        self._last_check_time = 0.0
        self._changes_detected = 0

        # 如果是事件模式，尝试加载pyinotify
        self._inotify_available = False
        if mode in [WatchMode.EVENT, WatchMode.HYBRID]:
            try:
                import pyinotify
                self._inotify_available = True
                self._setup_inotify()
                logger.info("Using pyinotify for file watching")
            except ImportError:
                logger.warning(
                    "pyinotify not available, falling back to polling mode")
                self.mode = WatchMode.POLL

    def _setup_inotify(self) -> None:
        """设置inotify监视（如果可用）"""
        if not self._inotify_available:
            return

        try:
            import pyinotify

            # 将在实际添加监视时配置
            self._wm = pyinotify.WatchManager()

            # 定义事件处理器
            class EventHandler(pyinotify.ProcessEvent):
                def __init__(self, watcher):
                    self.watcher = watcher

                def process_default(self, event):
                    # 只处理关注的事件类型
                    if not event.pathname or not os.path.exists(event.pathname):
                        return

                    if event.maskname in ['IN_MODIFY', 'IN_MOVED_TO',
                                          'IN_CREATE']:
                        # 将变更添加到队列
                        self.watcher._change_queue.put(event.pathname)

            # 创建通知器
            self._notifier = pyinotify.Notifier(self._wm, EventHandler(self))

            # 在后台线程处理事件
            self._notifier_thread = threading.Thread(
                target=self._process_inotify_events,
                daemon=True
            )

            # 需要监视的事件
            self._inotify_mask = pyinotify.IN_MODIFY | pyinotify.IN_MOVED_TO | pyinotify.IN_CREATE

        except Exception as e:
            logger.error(f"Error setting up inotify: {str(e)}")
            self._inotify_available = False
            self.mode = WatchMode.POLL

    def _process_inotify_events(self) -> None:
        """处理inotify事件的线程"""
        try:
            while not self._should_stop.is_set():
                # 处理等待的事件（非阻塞）
                if self._notifier.check_events(timeout=1000):
                    self._notifier.read_events()
                    self._notifier.process_events()
        except Exception as e:
            logger.error(f"Error processing inotify events: {str(e)}")
        finally:
            # 清理
            try:
                self._notifier.stop()
            except:
                pass

    def _add_inotify_watch(self, path: str) -> None:
        """添加inotify监视

        Args:
            path: 要监视的路径
        """
        if not self._inotify_available:
            return

        try:
            import pyinotify

            # 如果是文件，监视其所在目录
            watch_path = os.path.dirname(path) if os.path.isfile(path) else path

            # 添加监视
            self._wm.add_watch(watch_path, self._inotify_mask, rec=False)
            logger.debug(f"Added inotify watch for {watch_path}")
        except Exception as e:
            logger.error(f"Error adding inotify watch for {path}: {str(e)}")

    def watch_file(self, file_path: str,
                   callback: Optional[Callable[[str], None]] = None) -> bool:
        """监视单个配置文件

        Args:
            file_path: 文件路径
            callback: 文件变更回调函数

        Returns:
            bool: 是否成功添加监视
        """
        if not os.path.isfile(file_path):
            logger.error(f"Cannot watch non-existent file: {file_path}")
            return False

        try:
            # 计算文件哈希
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                return False

            # 添加到监视列表
            self._watched_files[file_path] = file_hash

            # 注册回调
            if callback:
                if file_path not in self._file_callbacks:
                    self._file_callbacks[file_path] = []
                self._file_callbacks[file_path].append(callback)

            # 如果使用事件模式，添加inotify监视
            if self.mode in [WatchMode.EVENT, WatchMode.HYBRID]:
                self._add_inotify_watch(file_path)

            logger.info(f"Started watching file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error watching file {file_path}: {str(e)}")
            return False

    def watch_directory(
            self,
            dir_path: str,
            pattern: Optional[str] = None,
            callback: Optional[Callable[[str], None]] = None,
            recursive: bool = False
    ) -> bool:
        """监视目录中的配置文件

        Args:
            dir_path: 目录路径
            pattern: 文件名模式
            callback: 文件变更回调函数
            recursive: 是否递归监视子目录

        Returns:
            bool: 是否成功添加监视
        """
        if not os.path.isdir(dir_path):
            logger.error(f"Cannot watch non-existent directory: {dir_path}")
            return False

        try:
            import re

            # 编译模式
            compiled_pattern = re.compile(pattern) if pattern else None

            # 获取初始文件列表和哈希
            dir_files = {}

            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    # 如果指定了模式，检查是否匹配
                    if compiled_pattern and not compiled_pattern.search(file):
                        continue

                    # 计算文件哈希
                    file_hash = self._get_file_hash(file_path)
                    if file_hash:
                        dir_files[file_path] = file_hash

                # 如果不递归，跳出循环
                if not recursive:
                    break

            # 添加到监视列表
            self._watched_dirs[dir_path] = dir_files

            # 注册回调
            if callback:
                if dir_path not in self._dir_callbacks:
                    self._dir_callbacks[dir_path] = []
                self._dir_callbacks[dir_path].append(callback)

            # 如果使用事件模式，添加inotify监视
            if self.mode in [WatchMode.EVENT, WatchMode.HYBRID]:
                self._add_inotify_watch(dir_path)
                if recursive:
                    for root, dirs, _ in os.walk(dir_path):
                        for subdir in dirs:
                            subdir_path = os.path.join(root, subdir)
                            self._add_inotify_watch(subdir_path)

            logger.info(
                f"Started watching directory: {dir_path} (files: {len(dir_files)})")
            return True
        except Exception as e:
            logger.error(f"Error watching directory {dir_path}: {str(e)}")
            return False

    def unwatch_file(self, file_path: str) -> bool:
        """停止监视文件

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否成功停止监视
        """
        if file_path in self._watched_files:
            del self._watched_files[file_path]
            if file_path in self._file_callbacks:
                del self._file_callbacks[file_path]
            logger.info(f"Stopped watching file: {file_path}")
            return True
        return False

    def unwatch_directory(self, dir_path: str) -> bool:
        """停止监视目录

        Args:
            dir_path: 目录路径

        Returns:
            bool: 是否成功停止监视
        """
        if dir_path in self._watched_dirs:
            del self._watched_dirs[dir_path]
            if dir_path in self._dir_callbacks:
                del self._dir_callbacks[dir_path]
            logger.info(f"Stopped watching directory: {dir_path}")
            return True
        return False

    def add_global_callback(self, callback: Callable[[str], None]) -> None:
        """添加全局文件变更回调

        Args:
            callback: 回调函数
        """
        if callback not in self._global_callbacks:
            self._global_callbacks.append(callback)

    def remove_global_callback(self, callback: Callable[[str], None]) -> bool:
        """移除全局文件变更回调

        Args:
            callback: 回调函数

        Returns:
            bool: 是否成功移除
        """
        if callback in self._global_callbacks:
            self._global_callbacks.remove(callback)
            return True
        return False

    def start(self) -> None:
        """开始监视"""
        if self._running:
            return

        self._should_stop.clear()

        # 启动通知线程
        self._notification_thread = threading.Thread(
            target=self._process_notifications,
            daemon=True
        )
        self._notification_thread.start()

        # 如果使用事件模式，启动inotify线程
        if self._inotify_available and self.mode in [WatchMode.EVENT,
                                                     WatchMode.HYBRID]:
            self._notifier_thread.start()

        # 如果使用轮询模式或混合模式，启动监视线程
        if self.mode in [WatchMode.POLL, WatchMode.HYBRID]:
            self._watch_thread = threading.Thread(
                target=self._watch_loop,
                daemon=True
            )
            self._watch_thread.start()

        self._running = True
        logger.info(f"Started config watcher in {self.mode} mode")

    def stop(self) -> None:
        """停止监视"""
        if not self._running:
            return

        logger.info("Stopping config watcher...")
        self._should_stop.set()

        # 等待线程结束
        if self._watch_thread:
            try:
                self._watch_thread.join(timeout=2.0)
            except:
                pass

        if self._notification_thread:
            try:
                # 添加一个空消息来确保通知线程能够退出
                self._change_queue.put(None)
                self._notification_thread.join(timeout=2.0)
            except:
                pass

        self._running = False
        logger.info("Config watcher stopped")

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        """获取文件的MD5哈希值

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 哈希值，如果失败则返回None
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                # 读取块而不是整个文件，以节省内存
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
            return file_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return None

    def _check_file_changes(self) -> List[str]:
        """检查文件变更

        Returns:
            List[str]: 变更的文件路径列表
        """
        changed_files = []

        # 检查单独监视的文件
        for file_path, stored_hash in list(self._watched_files.items()):
            # 如果文件不存在，跳过
            if not os.path.exists(file_path):
                continue

            # 检查哈希
            current_hash = self._get_file_hash(file_path)
            if not current_hash:
                continue

            if current_hash != stored_hash:
                # 更新存储的哈希
                self._watched_files[file_path] = current_hash
                changed_files.append(file_path)

        # 检查监视的目录
        for dir_path, files in list(self._watched_dirs.items()):
            # 如果目录不存在，跳过
            if not os.path.exists(dir_path):
                continue

            # 递归目录扫描所需参数
            recursive = True  # 假设是递归的，可以根据需要调整

            # 扫描目录
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)

                    # 检查哈希
                    current_hash = self._get_file_hash(file_path)
                    if not current_hash:
                        continue

                    if file_path in files:
                        # 已知文件，检查是否变更
                        if current_hash != files[file_path]:
                            files[file_path] = current_hash
                            changed_files.append(file_path)
                    else:
                        # 新文件
                        files[file_path] = current_hash
                        changed_files.append(file_path)

                # 如果不是递归模式，跳出循环
                if not recursive:
                    break

        return changed_files

    def _watch_loop(self) -> None:
        """监视循环"""
        logger.info(f"Watch loop started with interval {self.interval}s")

        while not self._should_stop.is_set():
            try:
                # 记录检查时间
                start_time = time.time()
                self._last_check_time = start_time

                # 检查文件变更
                changed_files = self._check_file_changes()

                # 如果有变更，通知回调
                for file_path in changed_files:
                    self._change_queue.put(file_path)
                    self._changes_detected += 1

                # 计算下一次检查的等待时间
                elapsed = time.time() - start_time
                wait_time = max(0.1, self.interval - elapsed)

                # 等待，但可以提前中断
                self._should_stop.wait(wait_time)
            except Exception as e:
                logger.error(f"Error in watch loop: {str(e)}")
                # 出错后等待较短时间再继续
                time.sleep(0.5)

    def _process_notifications(self) -> None:
        """处理通知队列"""
        while not self._should_stop.is_set():
            try:
                # 从队列获取变更文件
                file_path = self._change_queue.get(timeout=0.5)

                # 如果是None，说明是停止信号
                if file_path is None:
                    break

                # 调用相关回调
                self._notify_callbacks(file_path)

                # 标记任务完成
                self._change_queue.task_done()
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"Error processing notifications: {str(e)}")

    def _notify_callbacks(self, file_path: str) -> None:
        """通知回调函数

        Args:
            file_path: 变更的文件路径
        """
        logger.info(f"Detected change in config file: {file_path}")

        callbacks_to_call = []

        # 文件特定回调
        if file_path in self._file_callbacks:
            callbacks_to_call.extend(self._file_callbacks[file_path])

        # 目录回调
        for dir_path, callbacks in self._dir_callbacks.items():
            if file_path.startswith(dir_path):
                callbacks_to_call.extend(callbacks)

        # 全局回调
        callbacks_to_call.extend(self._global_callbacks)

        # 调用所有回调
        for callback in callbacks_to_call:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"Error in file change callback: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """获取监视统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "mode": self.mode,
            "running": self._running,
            "interval": self.interval,
            "watched_files": len(self._watched_files),
            "watched_dirs": len(self._watched_dirs),
            "total_files": len(self._watched_files) + sum(
                len(files) for files in self._watched_dirs.values()),
            "changes_detected": self._changes_detected,
            "last_check_time": self._last_check_time,
            "uptime": time.time() - self._last_check_time if self._last_check_time else 0
        }

    def get_watched_files(self) -> List[str]:
        """获取所有监视的文件

        Returns:
            List[str]: 文件路径列表
        """
        files = list(self._watched_files.keys())
        for dir_files in self._watched_dirs.values():
            files.extend(dir_files.keys())
        return files


# 获取监视器实例的工厂函数
def get_config_watcher(
        mode: str = WatchMode.POLL,
        interval: float = 1.0,
        auto_reload: bool = True
) -> ConfigWatcher:
    """获取配置文件监视器实例

    Args:
        mode: 监视模式
        interval: 检查间隔（秒）
        auto_reload: 是否自动重新加载配置

    Returns:
        ConfigWatcher: 监视器实例
    """
    return ConfigWatcher(mode, interval, auto_reload)


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建配置文件
    config_dir = "./config_test"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    test_config_path = os.path.join(config_dir, "test_config.json")
    with open(test_config_path, 'w') as f:
        f.write('{"test": "value", "number": 42}')

    # 创建监视器
    watcher = get_config_watcher(interval=0.5)


    # 定义回调函数
    def file_changed(file_path):
        print(f"File changed: {file_path}")
        # 读取并显示配置内容
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            print(f"New content: {content}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")


    # 添加监视
    watcher.watch_file(test_config_path, file_changed)
    watcher.watch_directory(config_dir, pattern=r"\.json$",
                            callback=lambda p: print(f"Directory change: {p}"))

    # 启动监视
    watcher.start()

    print(f"Watching file: {test_config_path}")
    print("Waiting for changes... (modify the file to see events)")
    print("Press Ctrl+C to exit")

    try:
        # 等待一会儿
        time.sleep(2)

        # 修改配置文件
        print("\nModifying config file...")
        with open(test_config_path, 'w') as f:
            f.write('{"test": "new value", "number": 100, "added": true}')

        # 等待检测变化
        time.sleep(2)

        # 创建新文件
        print("\nCreating new config file...")
        new_config_path = os.path.join(config_dir, "new_config.json")
        with open(new_config_path, 'w') as f:
            f.write('{"new": "config"}')

        # 等待检测变化
        time.sleep(2)

        # 显示统计信息
        stats = watcher.get_stats()
        print("\nWatcher stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 运行一会儿然后退出
        time.sleep(3)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # 停止监视
        watcher.stop()
        print("Watcher stopped")
