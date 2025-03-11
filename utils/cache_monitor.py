# -*- coding: utf-8 -*-
"""
缓存监控模块 - 管理和监控系统中的所有缓存

本模块提供:
1. 全局缓存监控和注册机制
2. 缓存使用情况分析和报告
3. 统计信息可视化和持久化
4. 定期缓存健康检查
"""
import os
import json
import time
import logging
import threading
import concurrent.futures
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 从日志配置导入日志记录器
from .logger_config import get_logger

logger = get_logger("CacheMonitor")


class CacheMonitor:
    """
    监控系统中所有缓存的工具

    提供缓存注册、统计分析、报告生成和健康检查功能
    """

    def __init__(self, stats_dir="logs/cache_stats"):
        """
        初始化缓存监控器

        Args:
            stats_dir: 统计信息存储目录
        """
        self.monitored_caches = {}
        self.stats_dir = stats_dir
        self._ensure_stats_dir()

        # 监控线程
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 300  # 5分钟

        # 监控历史
        self.history = {}
        self.history_length = 100

        logger.info("缓存监控器已初始化")

    def _ensure_stats_dir(self):
        """确保统计目录存在"""
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir, exist_ok=True)
            logger.info(f"已创建缓存统计目录: {self.stats_dir}")

    def register_cache(self, name, cache_instance):
        """
        注册一个缓存实例进行监控

        Args:
            name: 缓存名称
            cache_instance: 缓存实例

        Returns:
            bool: 是否成功注册
        """
        if cache_instance is None:
            logger.warning(f"尝试注册空缓存实例: {name}")
            return False

        if name in self.monitored_caches:
            logger.info(f"缓存 '{name}' 已注册，更新缓存实例")

        # 记录初始缓存属性
        cache_instance.monitor_register_time = time.time()
        self.monitored_caches[name] = cache_instance

        # 确保名称一致
        if hasattr(cache_instance, 'name'):
            if cache_instance.name != name:
                logger.warning(
                    f"缓存名称不一致: {cache_instance.name} vs {name}")
                cache_instance.name = name
        else:
            cache_instance.name = name

        capacity = getattr(cache_instance, 'capacity', 'unknown')
        logger.info(f"注册缓存: '{name}', 容量: {capacity}")

        # 初始化监控历史
        if name not in self.history:
            self.history[name] = []

        return True

    def unregister_cache(self, name):
        """
        取消注册缓存实例

        Args:
            name: 缓存名称

        Returns:
            bool: 是否成功取消注册
        """
        if name in self.monitored_caches:
            # 持久化最终统计信息
            self._persist_stats(name)
            del self.monitored_caches[name]
            logger.info(f"已取消注册缓存: '{name}'")
            return True
        return False

    def get_cache(self, name):
        """
        获取已注册的缓存实例

        Args:
            name: 缓存名称

        Returns:
            object: 缓存实例或None
        """
        return self.monitored_caches.get(name)

    def start_monitoring(self):
        """启动后台监控线程"""
        if self.monitoring:
            logger.warning("监控线程已在运行")
            return False

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop,
                                               daemon=True)
        self.monitor_thread.start()
        logger.info(f"缓存监控线程已启动，间隔: {self.monitor_interval}秒")
        return True

    def stop_monitoring(self):
        """停止后台监控线程"""
        if not self.monitoring:
            return False

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
            logger.info("缓存监控线程已停止")

        # 保存所有缓存的最终统计信息
        for name in self.monitored_caches:
            self._persist_stats(name)

        return True

    def _monitoring_loop(self):
        """监控线程主循环"""
        while self.monitoring:
            try:
                # 收集所有缓存的统计信息
                stats = self.get_all_stats()

                # 更新历史记录
                timestamp = time.time()
                for name, cache_stats in stats.items():
                    history_entry = {
                        "timestamp": timestamp,
                        "size": cache_stats.get("size", 0),
                        "hit_rate": float(
                            cache_stats.get("hit_rate", "0").rstrip("%")) / 100,
                        "usage_percent": float(
                            cache_stats.get("usage_percent", "0").rstrip(
                                "%")) / 100
                    }

                    if name in self.history:
                        self.history[name].append(history_entry)
                        # 限制历史长度
                        if len(self.history[name]) > self.history_length:
                            self.history[name] = self.history[name][
                                                 -self.history_length:]
                    else:
                        self.history[name] = [history_entry]

                # 执行健康检查
                self._check_cache_health(stats)

                # 每小时持久化统计信息
                if int(timestamp) % 3600 < self.monitor_interval:
                    self._persist_all_stats()

                # 等待下一个监控周期
                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"缓存监控循环出错: {e}")
                time.sleep(10)  # 错误后短暂等待

    def _check_cache_health(self, stats):
        """
        检查缓存健康状态

        Args:
            stats: 所有缓存的统计信息
        """
        issues = []

        for name, cache_stats in stats.items():
            # 检查使用率过高
            usage_str = cache_stats.get("usage_percent", "0%")
            usage = float(usage_str.rstrip("%")) / 100
            if usage > 0.9:  # 超过90%
                issues.append(f"缓存'{name}'使用率过高: {usage_str}")

            # 检查命中率过低
            hit_rate_str = cache_stats.get("hit_rate", "0%")
            hit_rate = float(hit_rate_str.rstrip("%")) / 100
            if hit_rate < 0.3 and hit_rate > 0:  # 低于30%，但有查询
                issues.append(f"缓存'{name}'命中率过低: {hit_rate_str}")

        # 记录发现的问题
        if issues:
            logger.warning("缓存健康检查发现问题:")
            for issue in issues:
                logger.warning(f"  - {issue}")

    def _persist_stats(self, cache_name):
        """
        持久化指定缓存的统计信息

        Args:
            cache_name: 缓存名称
        """
        if cache_name not in self.monitored_caches:
            return

        cache = self.monitored_caches[cache_name]

        try:
            from .cache_utils import persist_cache_stats
            persist_cache_stats(cache)
        except (ImportError, AttributeError):
            # 如果不支持标准持久化，使用内部方法
            try:
                stats = self._get_cache_stats(cache)
                stats_file = os.path.join(self.stats_dir,
                                          f"{cache_name}_stats.json")

                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2)

                logger.debug(f"已持久化缓存'{cache_name}'的统计信息")
            except Exception as e:
                logger.error(f"持久化缓存'{cache_name}'统计信息时出错: {e}")

    def _persist_all_stats(self):
        """持久化所有缓存的统计信息"""
        for name in self.monitored_caches:
            self._persist_stats(name)

    def _get_cache_stats(self, cache):
        """
        获取单个缓存的统计信息

        Args:
            cache: 缓存实例

        Returns:
            dict: 统计信息
        """
        try:
            # 标准化统计信息
            stats = {
                "name": getattr(cache, 'name', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "type": type(cache).__name__
            }

            # 大小和容量
            if hasattr(cache, 'values'):
                stats["size"] = len(cache.values)
            elif hasattr(cache, '__len__'):
                stats["size"] = len(cache)
            else:
                stats["size"] = 0

            stats["capacity"] = getattr(cache, 'capacity', 0)

            # 命中统计
            if hasattr(cache, 'hit_count') and hasattr(cache, 'miss_count'):
                stats["hit_count"] = cache.hit_count
                stats["miss_count"] = cache.miss_count
                total = cache.hit_count + cache.miss_count
                stats[
                    "hit_rate"] = f"{(cache.hit_count / total * 100) if total > 0 else 0:.2f}%"
            elif hasattr(cache, 'cache_hits') and hasattr(cache,
                                                          'cache_misses'):
                stats["hit_count"] = cache.cache_hits
                stats["miss_count"] = cache.cache_misses
                total = cache.cache_hits + cache.cache_misses
                stats[
                    "hit_rate"] = f"{(cache.cache_hits / total * 100) if total > 0 else 0:.2f}%"
            else:
                stats["hit_rate"] = "未知"

            # 使用率
            if stats["capacity"] > 0:
                stats[
                    "usage_percent"] = f"{(stats['size'] / stats['capacity'] * 100):.1f}%"
            else:
                stats["usage_percent"] = "未知"

            # 运行时间
            if hasattr(cache, 'monitor_register_time'):
                uptime = time.time() - cache.monitor_register_time
                stats["uptime"] = f"{uptime:.1f}秒"

            # 缓存特定统计信息
            if hasattr(cache, 'get_stats'):
                cache_stats = cache.get_stats()
                stats.update(cache_stats)

            return stats

        except Exception as e:
            logger.error(f"获取缓存统计信息时出错: {e}")
            return {"error": str(e)}

    def get_all_stats(self):
        """
        获取所有缓存的统计信息

        Returns:
            dict: 缓存名称到统计信息的映射
        """
        stats = {}
        for name, cache in self.monitored_caches.items():
            try:
                stats[name] = self._get_cache_stats(cache)
            except Exception as e:
                stats[name] = {'error': str(e)}

        return stats

    def print_stats(self, include_history=False):
        """
        打印所有缓存的统计信息

        Args:
            include_history: 是否包含历史统计

        Returns:
            dict: 统计信息
        """
        stats = self.get_all_stats()

        print("\n=== 缓存监控统计 ===")
        if not stats:
            print("未注册任何缓存!")
            return stats

        header_printed = False

        for name, cache_stats in sorted(stats.items()):
            if not header_printed:
                print(
                    f"{'名称':<20} {'大小':<10} {'容量':<10} {'使用率':<10} {'命中率':<10}")
                print("-" * 60)
                header_printed = True

            # 提取关键指标
            size = cache_stats.get("size", "?")
            capacity = cache_stats.get("capacity", "?")
            usage = cache_stats.get("usage_percent", "?")
            hit_rate = cache_stats.get("hit_rate", "?")

            print(
                f"{name:<20} {size:<10} {capacity:<10} {usage:<10} {hit_rate:<10}")

        print("\n详细统计:")
        for name, cache_stats in sorted(stats.items()):
            print(f"· {name}:")
            for key, value in sorted(cache_stats.items()):
                if key not in ["name", "timestamp", "type"]:
                    print(f"  - {key}: {value}")

            # 打印历史趋势（如果请求）
            if include_history and name in self.history and self.history[name]:
                history = self.history[name]
                if len(history) >= 2:
                    first = history[0]
                    last = history[-1]
                    duration = last["timestamp"] - first["timestamp"]

                    if duration > 0:
                        hit_rate_change = (last["hit_rate"] - first[
                            "hit_rate"]) / duration * 3600
                        usage_change = (last["usage_percent"] - first[
                            "usage_percent"]) / duration * 3600

                        print(f"  - 命中率趋势: {hit_rate_change:+.2f}%/小时")
                        print(f"  - 使用率趋势: {usage_change:+.2f}%/小时")

            print("")

        # 显示系统级别建议
        self._print_recommendations(stats)

        return stats

    def _print_recommendations(self, stats):
        """基于统计打印优化建议"""
        recommendations = []

        # 检查使用率过低的缓存
        for name, cache_stats in stats.items():
            usage_str = cache_stats.get("usage_percent", "0%")
            usage = float(usage_str.rstrip("%")) / 100
            capacity = cache_stats.get("capacity", 0)

            if usage < 0.3 and capacity > 50:  # 不到30%，容量大于50
                recommendations.append(
                    f"考虑减小缓存'{name}'的容量，当前使用率仅为{usage_str}")

            # 检查命中率异常的缓存
            hit_rate_str = cache_stats.get("hit_rate", "0%")
            hit_rate = float(hit_rate_str.rstrip("%")) / 100

            if 0 < hit_rate < 0.4:  # 命中率低于40%
                recommendations.append(
                    f"缓存'{name}'的命中率较低({hit_rate_str})，考虑优化缓存键生成或调整缓存项超时")

        # 输出建议
        if recommendations:
            print("\n=== 优化建议 ===")
            for i, recommendation in enumerate(recommendations, 1):
                print(f"{i}. {recommendation}")

    def generate_stats_report(self, output_dir=None):
        """
        生成缓存统计报告，包括图表

        Args:
            output_dir: 输出目录，默认为stats_dir

        Returns:
            str: 报告文件路径
        """
        if output_dir is None:
            output_dir = self.stats_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 生成报告时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"cache_report_{timestamp}.html")

        try:
            # 获取最新统计信息
            stats = self.get_all_stats()

            # 生成HTML报告
            with open(report_file, 'w', encoding='utf-8') as f:
                # 写入HTML头部
                f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>缓存系统报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .chart { width: 100%; max-width: 800px; height: 400px; margin: 20px 0; }
        .summary { background-color: #eef; padding: 10px; border-radius: 5px; }
        .recommendations { background-color: #fee; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>缓存系统性能报告</h1>
    <p>生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    <div class="summary">
        <h2>系统摘要</h2>
        <p>监控的缓存数量: """ + str(len(stats)) + """</p>
    </div>
""")

                # 写入缓存表格
                f.write("""
    <h2>缓存概览</h2>
    <table>
        <tr>
            <th>名称</th>
            <th>大小</th>
            <th>容量</th>
            <th>使用率</th>
            <th>命中率</th>
            <th>状态</th>
        </tr>
""")

                for name, cache_stats in sorted(stats.items()):
                    size = cache_stats.get("size", "?")
                    capacity = cache_stats.get("capacity", "?")
                    usage = cache_stats.get("usage_percent", "?")
                    hit_rate = cache_stats.get("hit_rate", "?")

                    # 确定状态
                    status = "正常"
                    hit_rate_val = float(
                        hit_rate.rstrip("%")) / 100 if isinstance(hit_rate,
                                                                  str) and hit_rate.endswith(
                        "%") else 0
                    usage_val = float(usage.rstrip("%")) / 100 if isinstance(
                        usage, str) and usage.endswith("%") else 0

                    if hit_rate_val < 0.3 and hit_rate_val > 0:
                        status = "命中率低"
                    elif usage_val > 0.9:
                        status = "使用率高"

                    f.write(f"""
        <tr>
            <td>{name}</td>
            <td>{size}</td>
            <td>{capacity}</td>
            <td>{usage}</td>
            <td>{hit_rate}</td>
            <td>{status}</td>
        </tr>""")

                f.write("""
    </table>
""")

                # 写入详细统计
                f.write("""
    <h2>详细统计</h2>
""")

                for name, cache_stats in sorted(stats.items()):
                    f.write(f"""
    <h3>{name}</h3>
    <table>
        <tr>
            <th>指标</th>
            <th>值</th>
        </tr>
""")

                    for key, value in sorted(cache_stats.items()):
                        if key not in ["name", "timestamp", "type"]:
                            f.write(f"""
        <tr>
            <td>{key}</td>
            <td>{value}</td>
        </tr>""")

                    f.write("""
    </table>
""")

                # 生成图表（如果有历史数据）
                if self.history:
                    # 为每个缓存生成使用率和命中率图表
                    for name, history in self.history.items():
                        if len(history) < 2:
                            continue

                        # 准备数据
                        timestamps = [entry["timestamp"] for entry in history]
                        hit_rates = [entry["hit_rate"] * 100 for entry in
                                     history]  # 转为百分比
                        usage_rates = [entry["usage_percent"] * 100 for entry in
                                       history]  # 转为百分比

                        # 为差距较大的时间范围提供时间轴标签
                        time_range = max(timestamps) - min(timestamps)

                        # 如果历史记录时间跨度大于1小时，使用小时:分钟格式
                        if time_range > 3600:
                            time_labels = [
                                datetime.fromtimestamp(ts).strftime("%H:%M") for
                                ts in timestamps]
                        else:
                            time_labels = [
                                f"{int((ts - min(timestamps)) / 60)}m" for ts in
                                timestamps]

                        # 写入图表容器
                        f.write(f"""
    <h3>{name} - 性能趋势</h3>
    <div class="chart-container">
        <canvas id="chart_{name}" class="chart"></canvas>
    </div>
    <script>
        var ctx = document.getElementById('chart_{name}').getContext('2d');
        var chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(time_labels)},
                datasets: [
                    {{
                        label: '命中率 (%)',
                        data: {json.dumps(hit_rates)},
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        pointRadius: 2,
                        borderWidth: 2,
                        yAxisID: 'y-axis-1'
                    }},
                    {{
                        label: '使用率 (%)',
                        data: {json.dumps(usage_rates)},
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        pointRadius: 2,
                        borderWidth: 2,
                        yAxisID: 'y-axis-2'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: '时间'
                        }}
                    }},
                    'y-axis-1': {{
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: '命中率 (%)'
                        }},
                        min: 0,
                        max: 100
                    }},
                    'y-axis-2': {{
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: '使用率 (%)'
                        }},
                        min: 0,
                        max: 100,
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }}
                }}
            }}
        }});
    </script>
""")

                # 写入推荐
                recommendations = []

                for name, cache_stats in stats.items():
                    usage_str = cache_stats.get("usage_percent", "0%")
                    usage = float(usage_str.rstrip("%")) / 100
                    capacity = cache_stats.get("capacity", 0)

                    if usage < 0.3 and capacity > 50:
                        recommendations.append(
                            f"考虑减小缓存'{name}'的容量，当前使用率仅为{usage_str}")

                    hit_rate_str = cache_stats.get("hit_rate", "0%")
                    hit_rate = float(hit_rate_str.rstrip("%")) / 100

                    if 0 < hit_rate < 0.4:
                        recommendations.append(
                            f"缓存'{name}'的命中率较低({hit_rate_str})，考虑优化缓存键生成或调整缓存项超时")

                if recommendations:
                    f.write("""
    <div class="recommendations">
        <h2>优化建议</h2>
        <ul>
""")

                    for recommendation in recommendations:
                        f.write(f"""
            <li>{recommendation}</li>""")

                    f.write("""
        </ul>
    </div>
""")

                # 引入Chart.js
                f.write("""
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</body>
</html>
""")

            logger.info(f"已生成缓存统计报告: {report_file}")
            return report_file

        except Exception as e:
            logger.error(f"生成缓存统计报告时出错: {e}")
            return None

    def generate_cache_charts(self, cache_name=None, output_dir=None):
        """
        生成缓存性能图表

        Args:
            cache_name: 指定缓存名称，None表示所有缓存
            output_dir: 输出目录，默认为stats_dir

        Returns:
            list: 生成的图表文件路径列表
        """
        try:
            import matplotlib.pyplot as plt

            if output_dir is None:
                output_dir = self.stats_dir

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            chart_files = []

            # 确定要处理的缓存
            caches_to_process = [
                cache_name] if cache_name else self.history.keys()

            for name in caches_to_process:
                if name not in self.history or len(self.history[name]) < 2:
                    continue

                history = self.history[name]

                # 准备数据
                timestamps = [entry["timestamp"] for entry in history]
                hit_rates = [entry["hit_rate"] * 100 for entry in
                             history]  # 转为百分比
                usage_rates = [entry["usage_percent"] * 100 for entry in
                               history]  # 转为百分比

                # 转换时间戳为可读格式
                time_labels = [datetime.fromtimestamp(ts).strftime("%H:%M") for
                               ts in timestamps]

                # 创建图表
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                               sharex=True)

                # 命中率图表
                ax1.plot(range(len(timestamps)), hit_rates, 'b-',
                         label='命中率')
                ax1.set_ylabel('命中率 (%)')
                ax1.set_ylim(0, 100)
                ax1.set_title(f"缓存 '{name}' 性能指标")
                ax1.legend()
                ax1.grid(True)

                # 使用率图表
                ax2.plot(range(len(timestamps)), usage_rates, 'r-',
                         label='使用率')
                ax2.set_ylabel('使用率 (%)')
                ax2.set_ylim(0, 100)
                ax2.set_xlabel('时间')
                ax2.legend()
                ax2.grid(True)

                # 设置x轴标签
                if len(time_labels) > 20:
                    # 如果数据点太多，只显示部分标签
                    step = len(time_labels) // 10
                    indices = range(0, len(time_labels), step)
                    ax2.set_xticks(indices)
                    ax2.set_xticklabels([time_labels[i] for i in indices],
                                        rotation=45)
                else:
                    ax2.set_xticks(range(len(time_labels)))
                    ax2.set_xticklabels(time_labels, rotation=45)

                plt.tight_layout()

                # 保存图表
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chart_file = os.path.join(output_dir,
                                          f"cache_{name}_chart_{timestamp}.png")
                plt.savefig(chart_file)
                plt.close(fig)

                chart_files.append(chart_file)
                logger.info(f"已生成缓存'{name}'的性能图表: {chart_file}")

            return chart_files

        except Exception as e:
            logger.error(f"生成缓存图表时出错: {e}")
            return []

    def clear_all_caches(self):
        """
        清除所有已注册缓存的内容

        Returns:
            int: 已清除的缓存数量
        """
        count = 0
        for name, cache in self.monitored_caches.items():
            try:
                # 保存最终统计信息
                self._persist_stats(name)

                # 清除缓存内容
                if hasattr(cache, 'clear'):
                    cache.clear()
                    count += 1
                    logger.info(f"已清除缓存: '{name}'")
                else:
                    logger.warning(f"缓存'{name}'没有clear方法")
            except Exception as e:
                logger.error(f"清除缓存'{name}'时出错: {e}")

        return count

    def analyze_cache_performance(self, cache_name=None):
        """
        分析缓存性能，提供详细报告

        Args:
            cache_name: 指定缓存名称，None表示所有缓存

        Returns:
            dict: 分析结果
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "caches": {}
        }

        # 确定要分析的缓存
        caches_to_analyze = [
            cache_name] if cache_name else self.monitored_caches.keys()

        for name in caches_to_analyze:
            if name not in self.monitored_caches:
                continue

            cache = self.monitored_caches[name]
            cache_stats = self._get_cache_stats(cache)

            # 计算基本指标
            usage_str = cache_stats.get("usage_percent", "0%")
            usage = float(usage_str.rstrip("%")) / 100

            hit_rate_str = cache_stats.get("hit_rate", "0%")
            hit_rate = float(hit_rate_str.rstrip("%")) / 100

            # 分析历史趋势
            trend_analysis = {}
            if name in self.history and len(self.history[name]) >= 2:
                history = self.history[name]
                first = history[0]
                last = history[-1]
                duration = last["timestamp"] - first["timestamp"]

                if duration > 0:
                    hit_rate_change = (last["hit_rate"] - first[
                        "hit_rate"]) / duration * 3600
                    usage_change = (last["usage_percent"] - first[
                        "usage_percent"]) / duration * 3600

                    trend_analysis = {
                        "hit_rate_trend": f"{hit_rate_change:+.2f}%/小时",
                        "usage_trend": f"{usage_change:+.2f}%/小时",
                        "duration_hours": duration / 3600
                    }

            # 评估缓存效率
            efficiency_score = hit_rate * 0.7 + (
                        1 - abs(usage - 0.7)) * 0.3  # 理想使用率约70%
            efficiency_rating = "优秀" if efficiency_score > 0.8 else "良好" if efficiency_score > 0.6 else "一般" if efficiency_score > 0.4 else "不佳"

            # 生成建议
            recommendations = []
            if hit_rate < 0.4 and hit_rate > 0:
                recommendations.append("优化缓存键生成算法，提高命中率")

            if usage < 0.3:
                recommendations.append("考虑减小缓存容量，提高内存效率")
            elif usage > 0.9:
                recommendations.append("考虑增加缓存容量，减少缓存替换频率")

            # 整合分析结果
            result["caches"][name] = {
                "basic_stats": cache_stats,
                "trend_analysis": trend_analysis,
                "efficiency": {
                    "score": efficiency_score,
                    "rating": efficiency_rating
                },
                "recommendations": recommendations
            }

        return result


# 单例实例
_monitor = CacheMonitor()


def get_monitor():
    """获取缓存监控器实例"""
    return _monitor
