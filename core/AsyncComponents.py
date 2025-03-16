# -*- coding: utf-8 -*-
"""
异步处理组件模块 - 提供异步视频处理管道
整合并封装异步处理相关的类，支持多线程处理
"""
import cv2
import time
import threading
import queue
import logging
import numpy as np
from utils.data_structures import CircularBuffer
from utils.cache_utils import create_standard_cache, generate_cache_key
from utils.logger_config import setup_logger, init_root_logger, \
    setup_utf8_console
from core.component_interface import BaseLifecycleComponent

logger = setup_logger("AsyncComponents")

class FrameQueue:
    """
    帧队列 - 专门为视频帧设计的高效队列
    基于CircularBuffer实现，自动管理内存和帧清理
    """

    def __init__(self, maxsize=5, event_system=None):
        """
        初始化帧队列

        Args:
            maxsize: 最大队列大小，超过此大小时自动丢弃最早的帧
            event_system: 事件系统实例，可选
        """
        self.max_size = maxsize
        self.queue = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.unfinished_tasks = 0
        self.all_tasks_done = threading.Condition(self.lock)

        # 性能统计
        self.put_count = 0
        self.get_count = 0
        self.drop_count = 0
        self.peak_size = 0

        # 保存事件系统
        self.events = event_system

        logger.info(f"帧队列已初始化，最大容量: {maxsize}")

    def qsize(self):
        """获取队列当前大小"""
        with self.lock:
            return len(self.queue)

    def empty(self):
        """检查队列是否为空"""
        with self.lock:
            return not self.queue

    def full(self):
        """检查队列是否已满"""
        with self.lock:
            return len(self.queue) >= self.max_size

    def put(self, item, block=True, timeout=None):
        """
        将项目放入队列

        Args:
            item: 要放入队列的项目
            block: 如果队列已满，是否阻塞等待
            timeout: 阻塞等待的超时时间

        Raises:
            queue.Full: 如果队列已满且block=False，或等待超时
        """
        with self.not_full:
            if not block:
                if len(self.queue) >= self.max_size:
                    raise queue.Full
            elif timeout is not None:
                endtime = time.time() + timeout
                while len(self.queue) >= self.max_size:
                    remaining = endtime - time.time()
                    if remaining <= 0.0:
                        raise queue.Full
                    self.not_full.wait(remaining)
            else:
                while len(self.queue) >= self.max_size:
                    self.not_full.wait()

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

            # 更新统计
            self.put_count += 1
            if len(self.queue) > self.peak_size:
                self.peak_size = len(self.queue)

            # 发布队列状态变化事件
            if self.events:
                self.events.publish("frame_queue_status", {
                    'action': 'put',
                    'size': len(self.queue),
                    'max_size': self.max_size,
                    'timestamp': time.time()
                })

    def put_nowait(self, item):
        """非阻塞方式放入项目"""
        return self.put(item, block=False)

    def get(self, block=True, timeout=None):
        """
        从队列获取项目

        Args:
            block: 如果队列为空，是否阻塞等待
            timeout: 阻塞等待的超时时间

        Returns:
            获取的项目

        Raises:
            queue.Empty: 如果队列为空且block=False，或等待超时
        """
        with self.not_empty:
            if not block:
                if not self.queue:
                    raise queue.Empty
            elif timeout is not None:
                endtime = time.time() + timeout
                while not self.queue:
                    remaining = endtime - time.time()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)
            else:
                while not self.queue:
                    self.not_empty.wait()

            item = self._get()
            self.not_full.notify()

            # 更新统计
            self.get_count += 1

            # 发布队列状态变化事件
            if self.events:
                self.events.publish("frame_queue_status", {
                    'action': 'get',
                    'size': len(self.queue),
                    'max_size': self.max_size,
                    'timestamp': time.time()
                })

            return item

    def get_nowait(self):
        """非阻塞方式获取项目"""
        return self.get(block=False)

    def task_done(self):
        """标记一个任务已完成"""
        with self.all_tasks_done:
            unfinished = self.unfinished_tasks - 1
            if unfinished < 0:
                raise ValueError('task_done() called too many times')
            self.unfinished_tasks = unfinished
            if unfinished == 0:
                self.all_tasks_done.notify_all()

    def join(self):
        """阻塞直到所有项目被处理完"""
        with self.all_tasks_done:
            while self.unfinished_tasks:
                self.all_tasks_done.wait()

    def _put(self, item):
        """内部方法：放入项目"""
        # 如果队列已满，移除最早的项目
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)  # 移除最早的帧
            self.drop_count += 1

            # 发布帧丢弃事件
            if self.events:
                self.events.publish("frame_dropped", {
                    'drop_count': self.drop_count,
                    'timestamp': time.time()
                })

        self.queue.append(item)

    def _get(self):
        """内部方法：获取项目"""
        return self.queue.pop(0)

    def clear(self):
        """清空队列"""
        with self.lock:
            self.queue.clear()
            self.not_full.notify_all()

            # 发布队列清空事件
            if self.events:
                self.events.publish("frame_queue_cleared", {
                    'timestamp': time.time()
                })

class VideoCapture:
    """视频捕获处理类 - 负责从相机获取帧并放入队列"""

    def __init__(self, camera, frame_queue, event_system=None):
        """
        初始化视频捕获器

        Args:
            camera: OpenCV视频捕获对象
            frame_queue: 帧队列，用于存储捕获的帧
            event_system: 事件系统实例，可选
        """
        self.camera = camera
        self.frame_queue = frame_queue
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        self.events = event_system

        # 性能控制参数
        self.max_fps = 0  # 0表示不限制，否则限制最大FPS
        self.last_frame_time = 0

        logger.info("视频捕获器初始化完成")

    def start(self):
        """启动捕获线程"""
        if self.thread is not None:
            logger.warning("视频捕获器已在运行")
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        self.start_time = time.time()
        self.frame_count = 0
        logger.info("视频捕获处理器已启动")
        print("视频捕获处理器已启动")

        # 发布视频捕获启动事件
        if self.events:
            self.events.publish("video_capture_started", {
                'timestamp': time.time()
            })

    def stop(self):
        """停止捕获线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            logger.info("视频捕获处理器已停止")
            print("视频捕获处理器已停止")

            # 发布视频捕获停止事件
            if self.events:
                self.events.publish("video_capture_stopped", {
                    'total_frames': self.frame_count,
                    'runtime': time.time() - self.start_time,
                    'timestamp': time.time()
                })

    def set_max_fps(self, max_fps):
        """设置最大帧率限制"""
        self.max_fps = max_fps
        logger.info(f"设置最大帧率限制: {max_fps} FPS")

        # 发布帧率设置事件
        if self.events:
            self.events.publish("fps_limit_set", {
                'max_fps': max_fps,
                'timestamp': time.time()
            })

    def get_fps(self):
        """获取当前估计帧率"""
        return self.fps

    def _capture_loop(self):
        """捕获循环 - 从相机获取帧并放入队列"""
        last_fps_update = time.time()
        min_frame_interval = 0  # 最小帧间隔（秒）

        while self.running:
            try:
                current_time = time.time()

                # 帧率限制
                if self.max_fps > 0:
                    min_frame_interval = 1.0 / self.max_fps
                    time_since_last_frame = current_time - self.last_frame_time
                    if time_since_last_frame < min_frame_interval:
                        # 休眠直到达到预期的帧间隔
                        sleep_time = min_frame_interval - time_since_last_frame
                        time.sleep(sleep_time)
                        current_time = time.time()

                # 获取帧
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("无法读取相机")
                    time.sleep(0.1)

                    # 发布相机读取失败事件
                    if self.events:
                        self.events.publish("camera_read_failed", {
                            'timestamp': time.time()
                        })

                    continue

                # 翻转帧以适应镜像效果
                frame = cv2.flip(frame, 1)
                self.last_frame_time = current_time

                # 帧率计算 (每秒更新一次)
                self.frame_count += 1
                if current_time - last_fps_update >= 1.0:
                    self.fps = self.frame_count / (
                            current_time - last_fps_update)
                    self.frame_count = 0
                    last_fps_update = current_time

                    # 发布帧率更新事件
                    if self.events:
                        self.events.publish("fps_updated", {
                            'fps': self.fps,
                            'timestamp': current_time
                        })

                # 创建帧数据包
                frame_data = {
                    'frame': frame,
                    'timestamp': current_time,
                    'index': self.frame_count,
                    'source': 'camera'
                }

                # 放入队列(非阻塞)
                try:
                    self.frame_queue.put_nowait(frame_data)

                    # 发布帧捕获成功事件
                    if self.events:
                        self.events.publish("frame_captured", {
                            'timestamp': current_time,
                            'frame_index': self.frame_count
                        })

                except queue.Full:
                    # 队列已满，丢弃最早的帧
                    try:
                        self.frame_queue.get_nowait()  # 移除一帧
                        self.frame_queue.put_nowait(frame_data)  # 添加新帧

                        # 发布帧替换事件
                        if self.events:
                            self.events.publish("frame_replaced", {
                                'timestamp': current_time,
                                'frame_index': self.frame_count
                            })

                    except:
                        pass
            except Exception as e:
                logger.error(f"视频捕获错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

                # 发布视频捕获错误事件
                if self.events:
                    self.events.publish("video_capture_error", {
                        'error': str(e),
                        'timestamp': time.time()
                    })

class DetectionProcessor:
    """检测处理类 - 负责对帧进行人体检测"""

    def __init__(self, detector, frame_queue, result_queue=None,
                 event_system=None):
        """
        初始化检测处理器

        Args:
            detector: 人体检测器对象
            frame_queue: 输入帧队列
            result_queue: 结果输出队列，如果为None则创建新队列
            event_system: 事件系统实例，可选
        """
        self.detector = detector
        self.frame_queue = frame_queue
        self.result_queue = result_queue or queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        self.started = False
        self.frame_skip = 0  # 跳过的帧数
        self.frame_counter = 0
        self.latest_detection = None
        self.detection_success_rate = 0.0  # 成功率统计
        self.detection_attempts = 0
        self.detection_successes = 0
        self.events = event_system

        # 如果检测器支持事件系统，传递事件系统实例
        if self.events and hasattr(self.detector, 'events'):
            self.detector.events = self.events

        # 添加缓存
        self.detection_cache = create_standard_cache(
            name="detection_processor",
            capacity=10,
            timeout=0.3,
            persistent=True
        )

        logger.info("检测处理器初始化完成")

    def start(self):
        """启动处理线程"""
        if self.thread is not None:
            logger.warning("检测处理器已在运行")
            return

        self.running = True
        self.started = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("检测处理器已启动")
        print("检测处理器已启动")

        # 发布检测处理器启动事件
        if self.events:
            self.events.publish("detection_processor_started", {
                'timestamp': time.time()
            })

    def stop(self):
        """停止处理线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            logger.info("检测处理器已停止")
            print("检测处理器已停止")

            # 发布检测处理器停止事件
            if self.events:
                self.events.publish("detection_processor_stopped", {
                    'timestamp': time.time(),
                    'success_rate': self.detection_success_rate
                })

    def set_frame_skip(self, skip_count):
        """设置帧跳过数量，用于降低处理频率"""
        self.frame_skip = max(0, skip_count)
        logger.info(f"检测处理器设置帧跳过: {skip_count}")

        # 发布帧跳过设置事件
        if self.events:
            self.events.publish("frame_skip_set", {
                'skip_count': skip_count,
                'timestamp': time.time()
            })

    def get_success_rate(self):
        """获取检测成功率"""
        return self.detection_success_rate

    def _process_loop(self):
        """处理循环 - 从帧队列获取数据并进行检测"""
        while self.running:
            try:
                # 获取帧(设置超时以便检查running标志)
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 帧计数与跳帧处理
                self.frame_counter += 1
                if self.frame_skip > 0 and self.frame_counter % (
                        self.frame_skip + 1) != 1:
                    continue

                # 处理帧
                frame = frame_data['frame']
                if frame is None or frame.size == 0:
                    continue

                # 创建缓存键
                small_frame = cv2.resize(frame, (16, 16))
                frame_hash = hash(str(small_frame.mean(axis=(0, 1))))

                # 检查缓存
                cached_detection = self.detection_cache.get(frame_hash)
                if cached_detection is not None:
                    # 更新时间戳
                    cached_detection['timestamp'] = frame_data['timestamp']

                    # 发布缓存命中事件
                    if self.events:
                        self.events.publish("detection_cache_hit", {
                            'timestamp': time.time(),
                            'frame_index': frame_data.get('index')
                        })

                    # 放入结果队列
                    try:
                        self.result_queue.put_nowait(cached_detection)
                    except queue.Full:
                        pass

                    continue

                # 人体姿态检测
                persons = []
                success = False
                try:
                    self.detection_attempts += 1
                    persons = self.detector.detect_pose(frame)
                    success = len(persons) > 0
                    if success:
                        self.detection_successes += 1

                        # 发布人体检测成功事件
                        if self.events:
                            self.events.publish("person_detection_success", {
                                'timestamp': time.time(),
                                'persons_count': len(persons),
                                'frame_index': frame_data.get('index')
                            })
                            # 在这里添加人体检测事件
                            if persons:
                                person = persons[0]
                                self.events.publish("person_detected", {
                                    'person': person,
                                    'confidence': person.get('confidence', 0.5),
                                    'frame': frame,
                                    'timestamp': time.time()
                                })
                    else:
                        # 发布人体检测失败事件
                        if self.events:
                            self.events.publish("person_detection_failure", {
                                'timestamp': time.time(),
                                'frame_index': frame_data.get('index')
                            })

                except Exception as e:
                    logger.error(f"检测错误: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

                    # 发布检测错误事件
                    if self.events:
                        self.events.publish("detection_error", {
                            'timestamp': time.time(),
                            'error': str(e),
                            'frame_index': frame_data.get('index')
                        })

                # 更新成功率统计
                if self.detection_attempts > 0:
                    self.detection_success_rate = self.detection_successes / self.detection_attempts

                    # 定期发布检测成功率事件
                    if self.detection_attempts % 50 == 0 and self.events:
                        self.events.publish("detection_success_rate_updated", {
                            'timestamp': time.time(),
                            'success_rate': self.detection_success_rate,
                            'attempts': self.detection_attempts,
                            'successes': self.detection_successes
                        })

                # 创建检测结果
                detection = {
                    'frame': frame,
                    'timestamp': frame_data['timestamp'],
                    'persons': persons,
                    'source': 'detection',
                    'success': success
                }

                # 保存最新检测
                self.latest_detection = detection

                # 缓存结果
                if success:
                    self.detection_cache.put(frame_hash, detection)

                # 放入结果队列(非阻塞)
                try:
                    self.result_queue.put_nowait(detection)
                except queue.Full:
                    # 队列已满，丢弃最早的结果
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(detection)

                        # 发布结果替换事件
                        if self.events:
                            self.events.publish("detection_result_replaced", {
                                'timestamp': time.time(),
                                'frame_index': frame_data.get('index')
                            })

                    except:
                        pass
            except Exception as e:
                logger.error(f"检测处理循环错误: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(0.01)  # 防止错误情况下CPU占用过高

                # 发布处理循环错误事件
                if self.events:
                    self.events.publish("detection_process_error", {
                        'timestamp': time.time(),
                        'error': str(e)
                    })

    def get_latest_detection(self):
        """获取最新的检测结果"""
        return self.latest_detection

class ActionRecognitionProcessor:
    """动作识别处理类 - 负责识别人体动作"""

    def __init__(self, recognizer, detection_queue, result_queue=None,
                 event_system=None):
        """
        初始化动作识别处理器

        Args:
            recognizer: 动作识别器对象
            detection_queue: 检测结果输入队列
            result_queue: 结果输出队列，如果为None则创建新队列
            event_system: 事件系统实例，可选
        """
        self.recognizer = recognizer
        self.detection_queue = detection_queue
        self.result_queue = result_queue or queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        self.latest_person = None
        self.latest_action = "Static"
        self.latest_result = None
        self.events = event_system

        # 如果识别器支持事件系统，传递事件系统实例
        if self.events and hasattr(self.recognizer, 'events'):
            self.recognizer.events = self.events

        logger.info("动作识别处理器初始化完成")

    def start(self):
        """启动处理线程"""
        if self.thread is not None:
            logger.warning("动作识别处理器已在运行")
            return

        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("动作识别处理器已启动")
        print("动作识别处理器已启动")

        # 发布动作识别处理器启动事件
        if self.events:
            self.events.publish("action_processor_started", {
                'timestamp': time.time()
            })

    def stop(self):
        """停止处理线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            logger.info("动作识别处理器已停止")
            print("动作识别处理器已停止")

            # 发布动作识别处理器停止事件
            if self.events:
                self.events.publish("action_processor_stopped", {
                    'timestamp': time.time(),
                    'last_action': self.latest_action
                })

    def _process_loop(self):
        """处理循环 - 从检测队列获取数据并进行动作识别"""
        while self.running:
            try:
                # 获取检测结果(设置超时以便检查running标志)
                try:
                    detection = self.detection_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 处理有效的检测结果
                if detection and 'persons' in detection and detection[
                    'persons']:
                    person = detection['persons'][0]
                    self.latest_person = person

                    # 识别动作
                    action_start = time.time()
                    action = self.recognizer.recognize_action(person)
                    action_time = time.time() - action_start

                    # 动作变化时发布事件
                    if self.events and action != self.latest_action:
                        self.events.publish("action_changed", {
                            'timestamp': time.time(),
                            'previous_action': self.latest_action,
                            'new_action': action,
                            'processing_time': action_time
                        })

                    self.latest_action = action

                    # 创建结果
                    result = {
                        'frame': detection.get('frame'),
                        'timestamp': detection.get('timestamp'),
                        'person': person,
                        'action': action,
                        'action_time': action_time,
                        'source': 'recognition'
                    }

                    # 保存最新结果
                    self.latest_result = result

                    # 放入结果队列(非阻塞)
                    try:
                        self.result_queue.put_nowait(result)

                        # 发布结果生成事件
                        if self.events:
                            self.events.publish("action_result_generated", {
                                'timestamp': time.time(),
                                'action': action,
                                'processing_time': action_time
                            })

                    except queue.Full:
                        # 队列已满，丢弃最早的结果
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)

                            # 发布结果替换事件
                            if self.events:
                                self.events.publish("action_result_replaced", {
                                    'timestamp': time.time(),
                                    'action': action
                                })

                        except:
                            pass
            except Exception as e:
                logger.error(f"动作识别处理错误: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(0.01)

                # 发布处理错误事件
                if self.events:
                    self.events.publish("action_process_error", {
                        'timestamp': time.time(),
                        'error': str(e)
                    })

    def get_latest_result(self):
        """获取最新动作识别结果"""
        if self.latest_result:
            return self.latest_result
        elif self.latest_person:
            return {
                'person': self.latest_person,
                'action': self.latest_action,
                'source': 'latest'
            }
        return None

class PositionMappingProcessor:
    """位置映射处理类 - 负责将检测结果映射到房间坐标系"""

    def __init__(self, mapper, action_queue, result_queue=None,
                 event_system=None):
        """
        初始化位置映射处理器

        Args:
            mapper: 位置映射器对象
            action_queue: 动作识别结果输入队列
            result_queue: 结果输出队列，如果为None则创建新队列
            event_system: 事件系统实例，可选
        """
        self.mapper = mapper
        self.action_queue = action_queue
        self.result_queue = result_queue or queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        self.latest_result = None
        self.frame_width = 640  # 默认值，会被更新
        self.frame_height = 480  # 默认值，会被更新
        self.events = event_system

        # 如果映射器支持事件系统，传递事件系统实例
        if self.events and hasattr(self.mapper, 'events'):
            self.mapper.events = self.events

        logger.info("位置映射处理器初始化完成")

    def start(self):
        """启动处理线程"""
        if self.thread is not None:
            logger.warning("位置映射处理器已在运行")
            return

        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("位置映射处理器已启动")
        print("位置映射处理器已启动")

        # 发布位置映射处理器启动事件
        if self.events:
            self.events.publish("position_processor_started", {
                'timestamp': time.time()
            })

    def set_frame_dimensions(self, width, height):
        """设置帧尺寸用于映射"""
        self.frame_width = width
        self.frame_height = height
        logger.info(f"位置映射器设置帧尺寸: {width}x{height}")

        # 发布帧尺寸设置事件
        if self.events:
            self.events.publish("frame_dimensions_set", {
                'timestamp': time.time(),
                'width': width,
                'height': height
            })

    def stop(self):
        """停止处理线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            logger.info("位置映射处理器已停止")
            print("位置映射处理器已停止")

            # 发布位置映射处理器停止事件
            if self.events:
                self.events.publish("position_processor_stopped", {
                    'timestamp': time.time()
                })

    def _process_loop(self):
        """处理循环 - 从动作队列获取结果并进行位置映射"""
        while self.running:
            try:
                # 获取动作结果
                try:
                    action_result = self.action_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # 处理有效的动作结果
                if action_result and 'person' in action_result:
                    person = action_result['person']
                    action = action_result.get('action', 'Static')
                    mapping_start = time.time()

                    # 映射位置到房间坐标
                    room_x, room_y, depth = self.mapper.map_position_to_room(
                        self.frame_width, self.frame_height,
                        self.mapper.room_width, self.mapper.room_height,
                        person
                    )

                    # 获取稳定位置
                    if hasattr(self.mapper, 'get_stable_position'):
                        stable_x, stable_y = self.mapper.get_stable_position(
                            room_x, room_y, action)
                    else:
                        stable_x, stable_y = room_x, room_y

                    # 计算平滑位置
                    if hasattr(self.mapper, 'smooth_position'):
                        smooth_x, smooth_y = self.mapper.smooth_position(
                            stable_x, stable_y)
                    else:
                        smooth_x, smooth_y = stable_x, stable_y

                    mapping_time = time.time() - mapping_start

                    # 发布位置映射事件
                    if self.events:
                        self.events.publish("position_mapped", {
                            'timestamp': time.time(),
                            'original_position': (room_x, room_y),
                            'stable_position': (stable_x, stable_y),
                            'smooth_position': (smooth_x, smooth_y),
                            'action': action,
                            'mapping_time': mapping_time
                        })

                    # 创建结果
                    result = {
                        'frame': action_result.get('frame'),
                        'timestamp': action_result.get('timestamp'),
                        'person': person,
                        'action': action,
                        'position': (smooth_x, smooth_y),
                        'stable_position': (stable_x, stable_y),
                        'depth': depth,
                        'room_position': (room_x, room_y),
                        'mapping_time': mapping_time,
                        'source': 'mapping'
                    }

                    # 保存最新结果
                    self.latest_result = result

                    # 放入结果队列
                    try:
                        self.result_queue.put_nowait(result)

                        # 发布结果生成事件
                        if self.events:
                            self.events.publish("position_result_generated", {
                                'timestamp': time.time(),
                                'position': (smooth_x, smooth_y),
                                'action': action
                            })

                    except queue.Full:
                        # 队列已满，丢弃最早的结果
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)

                            # 发布结果替换事件
                            if self.events:
                                self.events.publish("position_result_replaced",
                                                    {
                                                        'timestamp': time.time(),
                                                        'position': (
                                                        smooth_x, smooth_y)
                                                    })

                        except:
                            pass
            except Exception as e:
                logger.error(f"位置映射处理错误: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(0.01)

                # 发布处理错误事件
                if self.events:
                    self.events.publish("position_process_error", {
                        'timestamp': time.time(),
                        'error': str(e)
                    })

    def get_latest_result(self):
        """获取最新位置映射结果"""
        return self.latest_result

class AsyncPipeline(BaseLifecycleComponent):
    """完整异步管道类 - 协调各异步处理组件的工作"""

    def __init__(self, camera, detector, recognizer, mapper, system_manager=None,component_id="async_pipeline", component_type="Processor"):
        """
        初始化异步处理管道

        Args:
            camera: OpenCV视频捕获对象
            detector: 人体检测器对象
            recognizer: 动作识别器对象
            mapper: 位置映射器对象
            system_manager: 系统管理器对象，用于资源管理
        """
        # 初始化基础生命周期组件
        super().__init__(component_id, component_type)

        self.camera = camera
        self.detector = detector
        self.action_recognizer = recognizer
        self.position_mapper = mapper
        self.system_manager = system_manager

        # 检查是否有事件日志记录器
        self.event_logger = None
        if self.system_manager and hasattr(self.system_manager, 'event_logger'):
            self.event_logger = self.system_manager.event_logger
            logger.info("AsyncPipeline 已连接到事件日志记录器")
        # 保存系统管理器

        # 获取事件系统
        self.events = None
        if self.system_manager and hasattr(self.system_manager, 'event_logger'):
            self.events = self.system_manager.events

        # 创建队列
        self.frame_queue = FrameQueue(maxsize=5, event_system=self.events)
        self.detection_queue = queue.Queue(maxsize=5)
        self.action_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        # 创建处理组件
        self.video_capture = VideoCapture(camera, self.frame_queue, event_system=self.events)
        self.detection_processor = DetectionProcessor(detector,
                                                      self.frame_queue,
                                                      self.detection_queue,
                                                      event_system=self.events)
        self.action_processor = ActionRecognitionProcessor(recognizer,
                                                           self.detection_queue,
                                                           self.action_queue,
                                                           event_system=self.events)
        self.position_processor = PositionMappingProcessor(mapper,
                                                           self.action_queue,
                                                           self.result_queue,
                                                           event_system=self.events)

        # 配置参数
        self.performance_mode = 'balanced'  # 'high_speed', 'balanced', 'high_quality'
        self.started = False

        # 状态监控
        self.monitor_thread = None
        self.monitoring = False
        self.pipeline_stats = {
            'fps': 0,
            'detection_success_rate': 0,
            'latency': 0,  # 端到端延迟
            'frame_queue_size': 0,
            'detection_queue_size': 0,
            'action_queue_size': 0,
            'result_queue_size': 0,
            'stats_time': time.time()
        }

        # 传递事件系统到各组件
        if self.events:
            if hasattr(detector, 'events'):
                detector.events = self.events
            if hasattr(recognizer, 'events'):
                recognizer.events = self.events
            if hasattr(mapper, 'events'):
                mapper.events = self.events

        logger.info("异步处理管道初始化完成")

        # 发布管道初始化事件
        if self.events:
            self.events.publish("pipeline_initialized", {
                'timestamp': time.time(),
                'components': ['capture', 'detector', 'recognizer', 'mapper']
            })

    def _do_initialize(self) -> bool:
        """初始化异步管道"""
        try:
            # 只进行必要的初始化，但不启动线程
            return True
        except Exception as e:
            logger.error(f"初始化异步管道失败: {e}")
            return False

    def _do_start(self) -> bool:
        """启动异步管道"""
        try:
            # 启动异步管道
            if self.started:
                logger.warning("管道已经在运行")
                return True

            # 启动监控线程
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            # 启动处理组件
            self.video_capture.start()
            self.detection_processor.start()
            self.action_processor.start()
            self.position_processor.start()
            self.started = True

            logger.info("异步处理管道已启动")
            print("异步处理管道已启动")

            # 发布管道启动事件
            if self.events:
                self.events.publish("pipeline_started", {
                    'timestamp': time.time(),
                    'performance_mode': self.performance_mode
                })
            return True
        except Exception as e:
            logger.error(f"启动异步管道失败: {e}")
            return False

    def _do_pause(self) -> bool:
        """暂停异步管道"""
        # 对于异步管道，暂停可以是停止数据处理但保持线程运行的状态
        try:
            # 暂停可以实现为设置一个标志，让处理器暂时不处理新数据
            # 这里只是示例逻辑
            self.video_capture.set_max_fps(1)  # 降低帧率到最低
            logger.info("异步管道已暂停")
            return True
        except Exception as e:
            logger.error(f"暂停异步管道失败: {e}")
            return False

    def _do_stop(self) -> bool:
        """停止异步管道"""
        try:
            if not self.started:
                return True

            logger.info("正在停止管道...")
            print("正在停止管道...")

            # 停止监控线程
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
                self.monitor_thread = None

            # 按相反顺序停止处理组件
            self.position_processor.stop()
            self.action_processor.stop()

            # 清理检测处理器缓存
            if hasattr(self.detection_processor, 'detection_cache'):
                self.detection_processor.detection_cache.clear()
                logger.info("检测处理器缓存已清理")

            # 确保事件被记录
            if hasattr(self, 'event_logger') and self.event_logger:
                self.event_logger.save_current_events()

            self.detection_processor.stop()
            self.video_capture.stop()

            # 清空队列
            self._clear_queue(self.result_queue)
            self._clear_queue(self.action_queue)
            self._clear_queue(self.detection_queue)
            self._clear_queue(self.frame_queue)

            self.started = False
            logger.info("异步处理管道已停止")
            print("异步处理管道已停止")

            # 发布管道停止事件
            if self.events:
                self.events.publish("pipeline_stopped", {
                    'timestamp': time.time()
                })
            return True
        except Exception as e:
            logger.error(f"停止异步管道失败: {e}")
            return False

    def _do_destroy(self) -> bool:
        """销毁异步管道"""
        try:
            # 确保管道已停止
            if self.started:
                self._do_stop()

            # 释放资源
            self.camera = None
            self.detector = None
            self.action_recognizer = None
            self.position_mapper = None
            logger.info("异步管道资源已释放")
            return True
        except Exception as e:
            logger.error(f"销毁异步管道失败: {e}")
            return False

    def configure(self, frame_width, frame_height, performance_mode=None):
        """
        配置管道参数

        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
            performance_mode: 性能模式, 可选值: 'high_speed', 'balanced', 'high_quality'
        """
        # 设置帧尺寸
        self.position_processor.set_frame_dimensions(frame_width, frame_height)

        # 设置性能模式
        if performance_mode:
            self.set_performance_mode(performance_mode)

        logger.info(
            f"异步管道配置: {frame_width}x{frame_height}, 模式: {self.performance_mode}")

        # 发布管道配置事件
        if self.events:
            self.events.publish("pipeline_configured", {
                'timestamp': time.time(),
                'frame_width': frame_width,
                'frame_height': frame_height,
                'performance_mode': self.performance_mode
            })

    def set_performance_mode(self, mode):
        """
        设置性能模式

        Args:
            mode: 性能模式, 可选值: 'high_speed', 'balanced', 'high_quality'
        """
        valid_modes = ['high_speed', 'balanced', 'high_quality']
        if mode not in valid_modes:
            logger.warning(f"无效模式 '{mode}'，使用 'balanced'")
            mode = 'balanced'

        self.performance_mode = mode

        # 根据模式调整帧跳过
        if mode == 'high_speed':
            if hasattr(self.detection_processor, 'set_frame_skip'):
                self.detection_processor.set_frame_skip(2)  # 处理每3帧
            # 限制帧率
            self.video_capture.set_max_fps(15)
        elif mode == 'balanced':
            if hasattr(self.detection_processor, 'set_frame_skip'):
                self.detection_processor.set_frame_skip(1)  # 处理每2帧
            # 限制帧率
            self.video_capture.set_max_fps(25)
        else:  # high_quality
            if hasattr(self.detection_processor, 'set_frame_skip'):
                self.detection_processor.set_frame_skip(0)  # 处理每1帧
            # 不限制帧率
            self.video_capture.set_max_fps(0)

        logger.info(f"管道性能模式设置为 '{mode}'")
        print(f"管道性能模式设置为 '{mode}'")

        # 发布性能模式设置事件
        if self.events:
            self.events.publish("performance_mode_set", {
                'timestamp': time.time(),
                'mode': mode
            })

    def start(self):
        """启动所有处理线程"""
        if self.started:
            logger.warning("管道已经在运行")
            return

        # 启动监控线程
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # 启动处理组件
        self.video_capture.start()
        self.detection_processor.start()
        self.action_processor.start()
        self.position_processor.start()
        self.started = True

        logger.info("异步处理管道已启动")
        print("异步处理管道已启动")

        # 发布管道启动事件
        if self.events:
            self.events.publish("pipeline_started", {
                'timestamp': time.time(),
                'performance_mode': self.performance_mode
            })

    def get_result(self, timeout=None):
        """获取最新结果，支持超时参数"""
        return self.get_latest_result()

    def stop(self):
        """停止所有处理线程"""
        if not self.started:
            return

        logger.info("正在停止管道...")
        print("正在停止管道...")

        # 停止监控线程
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None

        # 按相反顺序停止处理组件
        self.position_processor.stop()
        self.action_processor.stop()

        # 清理检测处理器缓存
        if hasattr(self.detection_processor, 'detection_cache'):
            self.detection_processor.detection_cache.clear()
            logger.info("检测处理器缓存已清理")

        # 确保事件被记录
        if hasattr(self, 'event_logger') and self.event_logger:
            self.event_logger.save_current_events()

        self.detection_processor.stop()
        self.video_capture.stop()

        # 清空队列
        self._clear_queue(self.result_queue)
        self._clear_queue(self.action_queue)
        self._clear_queue(self.detection_queue)
        self._clear_queue(self.frame_queue)

        self.started = False
        logger.info("异步处理管道已停止")
        print("异步处理管道已停止")

        # 发布管道停止事件
        if self.events:
            self.events.publish("pipeline_stopped", {
                'timestamp': time.time()
            })

    def _clear_queue(self, queue_obj):
        """安全清空队列"""
        try:
            if hasattr(queue_obj, 'clear'):  # 如果队列有clear方法，使用它
                queue_obj.clear()
            else:  # 否则，尝试循环清空队列
                while True:
                    try:
                        queue_obj.get_nowait()
                    except queue.Empty:
                        break
        except Exception as e:
            logger.error(f"清空队列错误: {e}")
            if self.events:
                self.events.publish("queue_clear_error", {
                    'timestamp': time.time(),
                    'error': str(e)
                })

    def _monitor_loop(self):
        """监控循环 - 收集管道各个组件的性能统计"""
        while self.monitoring:
            try:
                current_time = time.time()

                # 收集性能统计
                self.pipeline_stats['fps'] = self.video_capture.get_fps()
                self.pipeline_stats[
                    'detection_success_rate'] = self.detection_processor.get_success_rate()
                self.pipeline_stats[
                    'frame_queue_size'] = self.frame_queue.qsize()
                self.pipeline_stats[
                    'detection_queue_size'] = self.detection_queue.qsize()
                self.pipeline_stats[
                    'action_queue_size'] = self.action_queue.qsize()
                self.pipeline_stats[
                    'result_queue_size'] = self.result_queue.qsize()
                self.pipeline_stats['stats_time'] = current_time

                # 发布性能统计更新事件
                if self.events:
                    self.events.publish("pipeline_stats_updated", {
                        'timestamp': current_time,
                        'stats': self.pipeline_stats
                    })

                # 如果有系统管理器，检查资源状态
                if self.system_manager:
                    resource_status = self.system_manager.check_resources()

                    # 根据资源状态自动调整性能模式
                    if hasattr(self.system_manager, 'should_apply_adaptation') and \
                            self.system_manager.should_apply_adaptation():
                        adaptation_level = self.system_manager.get_adaptation_level()

                        # 级别0=正常，1=警告，2=临界
                        if adaptation_level == 2:  # 资源紧张
                            if self.performance_mode != 'high_speed':
                                logger.warning("资源紧张，切换到高速模式")
                                self.set_performance_mode('high_speed')

                                # 发布资源适应事件
                                if self.events:
                                    self.events.publish("resource_adaptation", {
                                        'timestamp': current_time,
                                        'level': adaptation_level,
                                        'new_mode': 'high_speed'
                                    })
                        elif adaptation_level == 1:  # 资源警告
                            if self.performance_mode == 'high_quality':
                                logger.warning("资源警告，切换到平衡模式")
                                self.set_performance_mode('balanced')

                                # 发布资源适应事件
                                if self.events:
                                    self.events.publish("resource_adaptation", {
                                        'timestamp': current_time,
                                        'level': adaptation_level,
                                        'new_mode': 'balanced'
                                    })

                # 周期性记录性能统计
                if current_time % 30 < 0.1:  # 大约每30秒记录一次
                    logger.info(
                        f"管道性能: FPS={self.pipeline_stats['fps']:.1f}, "
                        f"检测成功率={self.pipeline_stats['detection_success_rate']:.2f}, "
                        f"队列大小: 帧={self.pipeline_stats['frame_queue_size']}, "
                        f"检测={self.pipeline_stats['detection_queue_size']}, "
                        f"动作={self.pipeline_stats['action_queue_size']}, "
                        f"结果={self.pipeline_stats['result_queue_size']}")

                time.sleep(0.5)  # 每0.5秒更新一次
            except Exception as e:
                logger.error(f"管道监控错误: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                time.sleep(1.0)

                # 发布监控错误事件
                if self.events:
                    self.events.publish("monitor_error", {
                        'timestamp': time.time(),
                        'error': str(e)
                    })

    def get_latest_result(self):
        """获取最新处理结果，优先使用最完整的数据"""
        try:
            # 1. 优先从结果队列获取完整处理结果
            try:
                result = self.result_queue.get_nowait()
                return result
            except queue.Empty:
                pass

            # 2. 使用位置处理器的最新结果
            position_result = self.position_processor.get_latest_result()
            if position_result:
                return position_result

            # 3. 使用动作处理器的最新结果
            action_result = self.action_processor.get_latest_result()
            if action_result:
                return action_result

            # 4. 使用检测处理器的最新结果
            detection = self.detection_processor.get_latest_detection()
            if detection and 'persons' in detection and detection['persons']:
                person = detection['persons'][0]
                return {
                    'frame': detection.get('frame'),
                    'person': person,
                    'action': 'Unknown',
                    'position': None,
                    'timestamp': detection.get('timestamp'),
                    'source': 'detection'
                }

            # 没有任何可用结果
            return None
        except Exception as e:
            logger.error(f"获取结果错误: {e}")
            import traceback
            logger.debug(traceback.format_exc())

            # 发布获取结果错误事件
            if self.events:
                self.events.publish("get_result_error", {
                    'timestamp': time.time(),
                    'error': str(e)
                })

            return None

    def get_pipeline_stats(self):
        """获取管道性能统计"""
        return self.pipeline_stats
