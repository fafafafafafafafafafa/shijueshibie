# utils/config.py
import os
import json
import logging


class AppConfig:
    """应用程序配置类 - 扩展版"""

    # 默认配置
    DEFAULT_CONFIG = {
        # 性能相关配置
        'performance_mode': 'balanced',
        'log_interval': 10,
        'memory_warning_threshold': 75,
        'memory_critical_threshold': 85,

        # 检测相关配置
        'use_mediapipe': False,
        'downscale_factor': 0.6,
        'keypoint_confidence_threshold': 0.5,

        # 动作识别配置
        'motion_cooldown': 0.5,
        'feature_extraction_method': 'standard',

        # 异步和线程配置
        'use_async': False,
        'thread_pool_size': 2,
        'processing_frame_skip': 1,

        # UI配置
        'window_width': 800,
        'window_height': 600,
        'show_debug_info': True,

        # 事件系统配置
        'event_history_size': 100,
        'event_logging_enabled': True
    }

    def __init__(self, **kwargs):
        """
        初始化配置

        Args:
            **kwargs: 键值对配置，优先级高于配置文件
        """
        # 从默认配置开始
        self._config = self.DEFAULT_CONFIG.copy()

        # 从环境变量加载
        self._load_from_env()

        # 从kwargs加载（最高优先级）
        self._config.update(kwargs)

        # 设置属性
        for key, value in self._config.items():
            setattr(self, key, value)

    def _load_from_env(self):
        """从环境变量加载配置"""
        prefix = "TRACKER_"
        for key in self.DEFAULT_CONFIG:
            env_key = prefix + key.upper()
            if env_key in os.environ:
                value = os.environ[env_key]
                # 尝试转换类型
                default_value = self.DEFAULT_CONFIG[key]
                try:
                    if isinstance(default_value, bool):
                        value = value.lower() in ('true', 'yes', '1')
                    else:
                        value = type(default_value)(value)
                    self._config[key] = value
                except ValueError:
                    logging.warning(
                        f"无法转换环境变量 {env_key} 的值 '{value}'")

    def _load_from_file(self, file_path):
        """从JSON配置文件加载"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._config.update(file_config)
        except Exception as e:
            logging.error(f"加载配置文件 {file_path} 失败: {e}")

    def save_to_file(self, file_path):
        """保存配置到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"保存配置文件 {file_path} 失败: {e}")
            return False

    def get(self, key, default=None):
        """获取配置项"""
        return self._config.get(key, default)

    def set(self, key, value):
        """设置配置项"""
        self._config[key] = value
        setattr(self, key, value)
        return True
