#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置持久化模块 - 处理配置的加载和保存

该模块提供了配置的持久化功能，支持从多种格式（JSON、YAML等）加载配置，
以及将配置保存到文件。它还提供了配置迁移和版本控制功能。
"""

import os
import json
import logging
import time
import shutil
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# 尝试导入YAML库，如果不可用则忽略
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning(
        "PyYAML not installed, YAML config files will not be supported")

# 尝试导入INI配置解析库
try:
    import configparser

    INI_AVAILABLE = True
except ImportError:
    INI_AVAILABLE = False
    logger.warning(
        "configparser not available, INI config files will not be supported")


class ConfigFormat:
    """配置文件格式枚举类"""
    JSON = "json"
    YAML = "yaml"
    INI = "ini"

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """获取支持的格式列表

        Returns:
            List[str]: 支持的格式列表
        """
        formats = [cls.JSON]
        if YAML_AVAILABLE:
            formats.append(cls.YAML)
        return formats


class ConfigPersistenceManager:
    """配置持久化管理器"""

    def __init__(
            self,
            config_dir: str = "config",
            default_format: str = ConfigFormat.JSON,
            enable_backups: bool = True,
            max_backups: int = 10
    ):
        """初始化配置持久化管理器

        Args:
            config_dir: 配置目录
            default_format: 默认配置格式
            enable_backups: 是否启用备份
            max_backups: 最大备份数量
        """
        self.config_dir = config_dir
        self.default_format = default_format
        self.enable_backups = enable_backups
        self.max_backups = max_backups
        self.backup_dir = os.path.join(config_dir, "backups")

        # 确保配置目录存在
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保配置目录和备份目录存在"""
        # 确保配置目录存在
        if not os.path.exists(self.config_dir):
            try:
                os.makedirs(self.config_dir)
                logger.info(f"Created config directory: {self.config_dir}")
            except OSError as e:
                logger.error(f"Failed to create config directory: {str(e)}")

        # 确保备份目录存在
        if self.enable_backups and not os.path.exists(self.backup_dir):
            try:
                os.makedirs(self.backup_dir)
                logger.info(f"Created backup directory: {self.backup_dir}")
            except OSError as e:
                logger.error(f"Failed to create backup directory: {str(e)}")

    def _detect_format(self, file_path: str) -> str:
        """检测配置文件格式

        Args:
            file_path: 文件路径

        Returns:
            str: 配置格式
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')

        if ext == 'json':
            return ConfigFormat.JSON
        elif ext in ('yaml', 'yml') and YAML_AVAILABLE:
            return ConfigFormat.YAML
        elif ext == 'ini':
            return ConfigFormat.INI

        # 默认假设为JSON
        return ConfigFormat.JSON

    def _create_backup(self, file_path: str) -> Optional[str]:
        """创建配置文件备份

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 备份文件路径，如果失败则返回None
        """
        if not self.enable_backups or not os.path.exists(file_path):
            return None

        try:
            # 获取文件名和扩展名
            file_name, file_ext = os.path.splitext(os.path.basename(file_path))

            # 生成备份文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{file_name}_{timestamp}{file_ext}"
            backup_path = os.path.join(self.backup_dir, backup_filename)

            # 复制文件到备份
            shutil.copy2(file_path, backup_path)

            logger.info(f"Created backup: {backup_path}")

            # 清理旧备份
            self._cleanup_old_backups(file_name, file_ext)

            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {str(e)}")
            return None

    def _cleanup_old_backups(self, base_name: str, extension: str) -> None:
        """清理旧备份文件

        Args:
            base_name: 文件基础名
            extension: 文件扩展名
        """
        if not os.path.exists(self.backup_dir):
            return

        try:
            # 查找匹配的备份文件
            pattern = re.compile(
                f"^{re.escape(base_name)}_\\d{{8}}_\\d{{6}}{re.escape(extension)}$")

            backups = []
            for filename in os.listdir(self.backup_dir):
                if pattern.match(filename):
                    file_path = os.path.join(self.backup_dir, filename)
                    # 获取修改时间
                    mod_time = os.path.getmtime(file_path)
                    backups.append((file_path, mod_time))

            # 按修改时间排序
            backups.sort(key=lambda x: x[1], reverse=True)

            # 删除多余的备份
            if len(backups) > self.max_backups:
                for file_path, _ in backups[self.max_backups:]:
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed old backup: {file_path}")
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove old backup {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {str(e)}")

    def generate_default_path(self, name: str,
                              format_: Optional[str] = None) -> str:
        """生成默认配置文件路径

        Args:
            name: 配置名称
            format_: 配置格式，如果为None则使用默认格式

        Returns:
            str: 配置文件路径
        """
        format_ = format_ or self.default_format
        extension = format_.lower()
        return os.path.join(self.config_dir, f"{name}.{extension}")

    def file_exists(self, file_path: str) -> bool:
        """检查配置文件是否存在

        Args:
            file_path: 文件路径

        Returns:
            bool: 文件是否存在
        """
        return os.path.exists(file_path)

    def load(self, file_path: str) -> Dict[str, Any]:
        """加载配置文件

        Args:
            file_path: 配置文件路径

        Returns:
            Dict[str, Any]: 加载的配置

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 配置格式不支持或解析错误
        """
        if not os.path.exists(file_path):
            logger.error(f"Config file not found: {file_path}")
            raise FileNotFoundError(f"Config file not found: {file_path}")

        format_ = self._detect_format(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if format_ == ConfigFormat.JSON:
                    config = json.load(f)
                elif format_ == ConfigFormat.YAML and YAML_AVAILABLE:
                    config = yaml.safe_load(f)
                elif format_ == ConfigFormat.INI and INI_AVAILABLE:
                    parser = configparser.ConfigParser()
                    parser.read_file(f)
                    # 将 INI 转换为嵌套字典
                    config = {section: dict(parser[section]) for section in
                              parser.sections()}
                    # 添加 DEFAULT 部分如果有值
                    if parser.defaults():
                        config['DEFAULT'] = dict(parser.defaults())
                else:
                    logger.error(f"Unsupported config format: {format_}")
                    raise ValueError(f"Unsupported config format: {format_}")

            logger.info(f"Loaded config from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {str(e)}")
            raise ValueError(f"Error parsing config file: {str(e)}")

    def save(
            self,
            config: Dict[str, Any],
            file_path: str,
            create_backup: bool = True,
            pretty: bool = True
    ) -> bool:
        """保存配置到文件

        Args:
            config: 配置字典
            file_path: 文件路径
            create_backup: 是否创建备份
            pretty: 是否美化输出

        Returns:
            bool: 是否成功保存
        """
        # 创建备份
        if create_backup and os.path.exists(file_path):
            self._create_backup(file_path)

        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                logger.error(
                    f"Failed to create directory {directory}: {str(e)}")
                return False

        format_ = self._detect_format(file_path)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format_ == ConfigFormat.JSON:
                    if pretty:
                        json.dump(config, f, indent=4, ensure_ascii=False)
                    else:
                        json.dump(config, f, ensure_ascii=False)
                elif format_ == ConfigFormat.YAML and YAML_AVAILABLE:
                    yaml.dump(config, f, default_flow_style=False)
                elif format_ == ConfigFormat.INI and INI_AVAILABLE:
                    parser = configparser.ConfigParser()
                    # 将嵌套字典转换为 INI 格式
                    for section, values in config.items():
                        if not isinstance(values, dict):
                            # 如果值不是字典，放到 DEFAULT 部分
                            parser['DEFAULT'][section] = str(values)
                            continue

                        parser[section] = {}
                        for key, value in values.items():
                            parser[section][key] = str(value)

                    parser.write(f)
                else:
                    logger.error(f"Unsupported config format: {format_}")
                    return False

            logger.info(f"Saved config to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {str(e)}")
            return False

    def merge(self, base_config: Dict[str, Any],
              override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并两个配置字典

        Args:
            base_config: 基础配置
            override_config: 覆盖配置

        Returns:
            Dict[str, Any]: 合并后的配置
        """
        result = base_config.copy()

        def _deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key],
                                                dict) and isinstance(value,
                                                                     dict):
                    target[key] = _deep_merge(target[key], value)
                else:
                    target[key] = value
            return target

        return _deep_merge(result, override_config)

    def get_backup_list(self, base_name: Optional[str] = None,
                        extension: Optional[str] = None) -> List[str]:
        """获取备份文件列表

        Args:
            base_name: 文件基础名，如果为None则返回所有备份
            extension: 文件扩展名，如果为None则不过滤扩展名

        Returns:
            List[str]: 备份文件路径列表
        """
        if not os.path.exists(self.backup_dir):
            return []

        backups = []

        try:
            for filename in os.listdir(self.backup_dir):
                file_path = os.path.join(self.backup_dir, filename)
                if not os.path.isfile(file_path):
                    continue

                if base_name and not filename.startswith(f"{base_name}_"):
                    continue

                if extension:
                    if not filename.endswith(extension):
                        continue

                backups.append(file_path)

            # 按修改时间排序
            backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            return backups
        except Exception as e:
            logger.error(f"Error getting backup list: {str(e)}")
            return []

    def restore_backup(self, backup_path: str,
                       target_path: Optional[str] = None) -> bool:
        """恢复配置备份

        Args:
            backup_path: 备份文件路径
            target_path: 目标文件路径，如果为None则使用原始文件路径

        Returns:
            bool: 是否成功恢复
        """
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_path}")
            return False

        # 如果未指定目标路径，尝试从备份文件名推断
        if target_path is None:
            filename = os.path.basename(backup_path)
            # 尝试从"name_20201231_235959.json"格式提取原始文件名
            match = re.match(r"^(.+?)_\d{8}_\d{6}(\..+)$", filename)
            if match:
                base_name, extension = match.groups()
                target_path = os.path.join(self.config_dir,
                                           f"{base_name}{extension}")
            else:
                logger.error(
                    f"Cannot determine target path for backup: {backup_path}")
                return False

        try:
            # 如果目标文件已存在，先备份它
            if os.path.exists(target_path):
                self._create_backup(target_path)

            # 复制备份到目标路径
            shutil.copy2(backup_path, target_path)

            logger.info(f"Restored backup from {backup_path} to {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {str(e)}")
            return False

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """获取文件的哈希值

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 文件哈希值，如果文件不存在则返回None
        """
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)

            return file_hash.hexdigest()
        except Exception as e:
            logger.error(
                f"Error calculating file hash for {file_path}: {str(e)}")
            return None

    def has_changed(self, file_path: str, previous_hash: str) -> bool:
        """检查文件是否已更改

        Args:
            file_path: 文件路径
            previous_hash: 先前的哈希值

        Returns:
            bool: 文件是否已更改
        """
        current_hash = self.get_file_hash(file_path)
        if current_hash is None:
            return True  # 如果文件不存在，则视为已更改

        return current_hash != previous_hash

    def export_to_format(self, config: Dict[str, Any], format_: str) -> \
    Optional[str]:
        """将配置导出为指定格式的字符串

        Args:
            config: 配置字典
            format_: 目标格式

        Returns:
            Optional[str]: 格式化后的字符串，如果格式不支持则返回None
        """
        try:
            if format_ == ConfigFormat.JSON:
                return json.dumps(config, indent=4, ensure_ascii=False)
            elif format_ == ConfigFormat.YAML and YAML_AVAILABLE:
                return yaml.dump(config, default_flow_style=False)
            else:
                logger.error(f"Unsupported export format: {format_}")
                return None
        except Exception as e:
            logger.error(f"Error exporting config to {format_}: {str(e)}")
            return None

    def import_from_string(self, config_str: str, format_: str) -> Optional[
        Dict[str, Any]]:
        """从字符串导入配置

        Args:
            config_str: 配置字符串
            format_: 配置格式

        Returns:
            Optional[Dict[str, Any]]: 导入的配置，如果导入失败则返回None
        """
        try:
            if format_ == ConfigFormat.JSON:
                return json.loads(config_str)
            elif format_ == ConfigFormat.YAML and YAML_AVAILABLE:
                return yaml.safe_load(config_str)
            else:
                logger.error(f"Unsupported import format: {format_}")
                return None
        except Exception as e:
            logger.error(f"Error importing config from string: {str(e)}")
            return None

    def scan_config_files(self, pattern: Optional[str] = None) -> List[str]:
        """扫描配置目录中的配置文件

        Args:
            pattern: 文件名匹配模式（正则表达式）

        Returns:
            List[str]: 配置文件路径列表
        """
        result = []

        if not os.path.exists(self.config_dir):
            return result

        try:
            compiled_pattern = re.compile(pattern) if pattern else None

            for filename in os.listdir(self.config_dir):
                file_path = os.path.join(self.config_dir, filename)

                # 跳过目录和备份目录
                if os.path.isdir(file_path) or filename == os.path.basename(
                        self.backup_dir):
                    continue

                # 如果指定了模式，检查是否匹配
                if compiled_pattern and not compiled_pattern.match(filename):
                    continue

                # 检查是否是支持的格式
                format_ = self._detect_format(file_path)
                if format_ in [ConfigFormat.JSON, ConfigFormat.YAML,
                               ConfigFormat.INI]:
                    result.append(file_path)

            return result
        except Exception as e:
            logger.error(f"Error scanning config directory: {str(e)}")
            return []
