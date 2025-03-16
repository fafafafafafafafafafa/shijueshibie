#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置集成模块 - 提供应用程序与配置系统之间的集成

该模块作为应用程序和配置系统之间的桥梁，定义了默认配置值，
并提供了配置系统的集成接口，简化应用程序与配置系统的交互。
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置

    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        "app": {
            "name": "人体姿态跟踪系统",
            "version": "1.0.0"
        },
        "detector": {
            "use_mediapipe": False,
            "performance_mode": "balanced",
            "downscale_factor": 0.6,
            "keypoint_confidence_threshold": 0.5,
            "roi_enabled": True,
            "roi_padding": 500,
            "roi_padding_factor": 2.0,
            "roi_padding_ratio": 1.0,
            "full_frame_interval": 2.0
        },
        "action_recognizer": {
            "keypoint_confidence_threshold": 0.5,
            "position_movement_threshold": 10,
            "motion_cooldown": 0.5,
            "history_length": 10,
            "action_history_length": 5,
            "waving_threshold": 50,
            "jumping_threshold": 30,
            "moving_threshold": 10,
            "frame_interval": 3,
            "enable_ml_model": False,
            "enable_dtw": False,
            "enable_threading": False
        },
        "position_mapper": {
            "room_width": 800,
            "room_height": 600,
            "min_height": 50,
            "max_height": 400,
            "depth_scale": 8.0,
            "backward_enabled": True,
            "backward_exponent": 2.2,
            "normal_exponent": 1.8,
            "use_smoothing": True,
            "max_occlusion_frames": 30
        },
        "system": {
            "log_interval": 10,
            "memory_warning_threshold": 75,
            "memory_critical_threshold": 85,
            "async_mode": False,
            "performance_mode": "balanced"
        },
        "visualizer": {
            "show_skeleton": True,
            "show_bounding_box": True,
            "show_keypoints": True,
            "show_labels": True,
            "max_trail_points": 50,
            "trail_fade": True,
            "room_width": 800,
            "room_height": 600,
            "background_color": [240, 240, 240],
            "person_color": [0, 120, 255],
            "text_color": [10, 10, 10],
            "trail_color": [0, 0, 255]
        },
        "ui": {
            "show_debug_info": True,
            "camera_width": 640,
            "camera_height": 480,
            "theme": "light",
            "font_size": 0.5,
            "font_thickness": 1,
            "show_fps": True,
            "show_feature_states": True
        },
        "calibration": {
            "auto_calibrate": True,
            "reference_height": 200,
            "calibration_frames": 30,
            "min_confidence": 0.7
        }
    }


def apply_config_to_component(component: Any, config: Dict[str, Any],
                              config_key: str) -> bool:
    """
    将配置应用到组件

    Args:
        component: 目标组件
        config: 配置字典
        config_key: 组件对应的配置键

    Returns:
        bool: 是否成功应用配置
    """
    if config_key not in config:
        logger.warning(f"配置中不存在键 '{config_key}'")
        return False

    component_config = config[config_key]

    try:
        # 如果组件有apply_config方法，使用它
        if hasattr(component, 'apply_config') and callable(
                component.apply_config):
            component.apply_config(component_config)
            logger.debug(f"已通过apply_config方法为组件应用'{config_key}'配置")
            return True

        # 如果组件有config属性，直接更新它
        elif hasattr(component, 'config'):
            if isinstance(component.config, dict):
                component.config.update(component_config)
                logger.debug(
                    f"已通过更新config属性为组件应用'{config_key}'配置")
                return True

        # 否则尝试直接设置属性
        else:
            applied = False
            for key, value in component_config.items():
                if hasattr(component, key):
                    setattr(component, key, value)
                    applied = True

            if applied:
                logger.debug(f"已通过设置属性为组件应用'{config_key}'配置")
                return True
            else:
                logger.warning(
                    f"无法为组件应用配置'{config_key}'：没有匹配的属性")
                return False

    except Exception as e:
        logger.error(f"应用配置'{config_key}'时出错: {e}")
        return False


def get_component_config(component: Any, config_key: str) -> Optional[
    Dict[str, Any]]:
    """
    获取组件当前配置

    Args:
        component: 目标组件
        config_key: 组件对应的配置键

    Returns:
        Optional[Dict[str, Any]]: 组件配置字典，如果获取失败则返回None
    """
    try:
        # 如果组件有get_config方法，使用它
        if hasattr(component, 'get_config') and callable(component.get_config):
            return component.get_config()

        # 如果组件有config属性，返回它的副本
        elif hasattr(component, 'config'):
            if isinstance(component.config, dict):
                return component.config.copy()

        # 否则尝试根据默认配置收集属性
        else:
            default_config = create_default_config()
            if config_key in default_config:
                result = {}
                for key in default_config[config_key].keys():
                    if hasattr(component, key):
                        result[key] = getattr(component, key)
                return result

        return None
    except Exception as e:
        logger.error(f"获取组件配置'{config_key}'时出错: {e}")
        return None


def get_default_feature_toggles() -> Dict[str, bool]:
    """
    获取默认功能开关状态

    Returns:
        Dict[str, bool]: 功能名称到开关状态的映射
    """
    config = create_default_config()
    return {
        "mediapipe": config["detector"]["use_mediapipe"],
        "ml_model": config["action_recognizer"]["enable_ml_model"],
        "dtw": config["action_recognizer"]["enable_dtw"],
        "threading": config["action_recognizer"]["enable_threading"],
        "async": config["system"]["async_mode"],
        "backward_enhancement": config["position_mapper"]["backward_enabled"],
        "debug_mode": config["ui"]["show_debug_info"]
    }


# 以下是使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 获取默认配置
    default_config = create_default_config()
    print("默认配置示例:")
    print(f"应用名称: {default_config['app']['name']}")
    print(f"检测器性能模式: {default_config['detector']['performance_mode']}")
    print(
        f"动作识别器历史长度: {default_config['action_recognizer']['history_length']}")

    # 获取默认功能开关
    feature_toggles = get_default_feature_toggles()
    print("\n默认功能开关状态:")
    for feature, state in feature_toggles.items():
        print(f"{feature}: {'开启' if state else '关闭'}")
