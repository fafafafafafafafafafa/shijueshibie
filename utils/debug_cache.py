#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
缓存调试工具 - 提供缓存系统检查和诊断功能

本工具可以:
1. 检查已注册的缓存
2. 启动缓存监控
3. 持久化缓存统计
4. 生成分析报告
"""

import os
import sys
import time


def setup_environment():
    """设置环境，确保导入路径正确"""
    # 获取当前文件目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 将当前目录添加到路径
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # 获取父目录并添加到路径
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # 尝试导入utils模块
    try:
        import utils
        # 确保utils模块的路径添加到系统路径
        utils_dir = os.path.dirname(utils.__file__)
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        print(f"已找到utils模块: {utils.__file__}")
    except ImportError:
        # 如果无法导入，尝试查找utils目录
        utils_dir = os.path.join(parent_dir, 'utils')
        if os.path.exists(utils_dir) and utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
            print(f"已添加utils目录到路径: {utils_dir}")

    # 确保日志目录存在
    ensure_log_dirs()


def ensure_log_dirs():
    """确保日志目录存在"""
    log_dirs = [
        "logs",
        "logs/cache_stats"
    ]

    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                print(f"已创建目录: {log_dir}")
            except Exception as e:
                print(f"创建目录 {log_dir} 时出错: {e}")


def run_debug():
    """运行缓存调试程序"""
    print("=== 缓存监控调试工具 ===")

    try:
        # 1. 导入监控器
        print("\n1. 获取缓存监控器...")

        try:
            from utils.cache_monitor import get_monitor
        except ImportError:
            try:
                # 尝试直接导入
                from cache_monitor import get_monitor
            except ImportError:
                print("无法导入缓存监控器，请检查路径设置")
                return

        monitor = get_monitor()

        # 2. 显示已注册缓存
        print("\n2. 检查已注册的缓存...")
        current_caches = monitor.monitored_caches
        print(f"已注册的缓存数量: {len(current_caches)}")

        if not current_caches:
            print("  - 未发现已注册的缓存")
        else:
            for name, cache in current_caches.items():
                print(
                    f"  - {name}: 大小={len(cache.values) if hasattr(cache, 'values') else '?'}, 容量={getattr(cache, 'capacity', '?')}")

        # 3. 启动监控
        if not monitor.monitoring:
            print("\n3. 启动缓存监控...")
            monitor.start_monitoring()
            print("  ✓ 监控线程已启动")
        else:
            print("\n3. 缓存监控已在运行")

        # 4. 立即持久化所有缓存统计
        print("\n4. 持久化缓存统计...")

        if hasattr(monitor, 'persist_all_stats'):
            # 使用新的公共方法
            count = monitor.persist_all_stats()
            print(f"  ✓ 已持久化 {count} 个缓存的统计信息")
        else:
            # 兼容旧版本，使用内部方法
            for name in current_caches:
                monitor._persist_stats(name)
            print("  ✓ 已持久化所有缓存统计")

        # 5. 生成分析报告
        print("\n5. 生成分析报告...")
        try:
            from utils.cache_analyzer import analyze_caches
        except ImportError:
            try:
                # 尝试直接导入
                from cache_analyzer import analyze_caches
            except ImportError:
                print("无法导入缓存分析器，请检查路径设置")
                return

        report_file = analyze_caches()
        if report_file:
            print(f"  ✓ 报告已生成: {report_file}")

            # 尝试显示报告内容预览
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    preview = f.read(500)  # 读取前500个字符作为预览
                    print("\n报告预览:")
                    print("-" * 50)
                    print(preview + "...")
                    print("-" * 50)
            except Exception as e:
                print(f"无法显示报告预览: {e}")
        else:
            print("  ✗ 报告生成失败，可能是未找到缓存统计数据")

        # 6. 执行健康检查
        print("\n6. 执行缓存健康检查...")
        try:
            stats = monitor.get_all_stats()
            issues_found = False

            for name, cache_stats in stats.items():
                # 检查使用率过高
                usage_str = cache_stats.get("usage_percent", "0%")
                usage = float(usage_str.rstrip("%")) / 100 if isinstance(
                    usage_str, str) else 0

                # 检查命中率过低
                hit_rate_str = cache_stats.get("hit_rate", "0%")
                hit_rate = float(hit_rate_str.rstrip("%")) / 100 if isinstance(
                    hit_rate_str, str) else 0

                # 输出问题
                if usage > 0.9:
                    print(f"  ! 警告: 缓存'{name}'使用率过高: {usage_str}")
                    issues_found = True

                if 0 < hit_rate < 0.3:
                    print(f"  ! 警告: 缓存'{name}'命中率过低: {hit_rate_str}")
                    issues_found = True

            if not issues_found:
                print("  ✓ 未发现缓存问题")
        except Exception as e:
            print(f"  ✗ 健康检查执行出错: {e}")

        print(
            "\n完成! 现在你可以在 logs/cache_stats/ 目录中查看缓存统计数据和分析报告")

    except Exception as e:
        print(f"执行缓存调试时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置环境
    setup_environment()

    # 运行调试
    run_debug()
