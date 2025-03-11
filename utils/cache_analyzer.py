# -*- coding: utf-8 -*-
"""
缓存分析器模块 - 提供缓存统计数据的分析和报告功能

本模块提供:
1. 缓存统计数据的加载和分析
2. 缓存性能趋势识别
3. 生成缓存优化建议
4. 创建分析报告
"""
import os
import json
import time
import glob
import traceback
from datetime import datetime
import logging
import tempfile
import numpy as np

# 配置日志记录器
try:
    from .logger_config import get_logger

    logger = get_logger("CacheAnalyzer")
except (ImportError, AttributeError):
    # 如果无法导入，创建基本日志记录器
    logger = logging.getLogger("CacheAnalyzer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class CacheAnalyzer:
    """
    缓存性能分析工具

    分析缓存统计数据并生成性能报告和优化建议
    """

    def __init__(self, stats_dir="logs/cache_stats"):
        """
        初始化缓存分析器

        Args:
            stats_dir: 统计数据目录
        """
        # 处理统计目录
        if stats_dir is None:
            stats_dir = "logs/cache_stats"
        elif not isinstance(stats_dir, str):
            logger.warning(
                f"无效的统计目录类型: {type(stats_dir)}，使用默认路径")
            stats_dir = "logs/cache_stats"

        self.stats_dir = stats_dir
        self.cache_stats = {}  # 按缓存名称组织的统计数据
        self.time_series = {}  # 各缓存的时间序列数据

        # 确保统计目录存在
        try:
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir, exist_ok=True)
                logger.info(f"已创建缓存统计目录: {stats_dir}")
        except Exception as e:
            logger.error(f"创建统计目录时出错: {e}")
            # 尝试使用相对路径
            alt_dir = os.path.join(os.getcwd(), "logs/cache_stats")
            logger.info(f"尝试使用备用路径: {alt_dir}")
            try:
                if not os.path.exists(alt_dir):
                    os.makedirs(alt_dir, exist_ok=True)
                self.stats_dir = alt_dir
                logger.info(f"使用备用统计目录: {alt_dir}")
            except Exception as e2:
                logger.error(f"创建备用统计目录时出错: {e2}")
                # 最后尝试在临时目录创建
                temp_dir = os.path.join(tempfile.gettempdir(), "cache_stats")
                try:
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir, exist_ok=True)
                    self.stats_dir = temp_dir
                    logger.info(f"使用临时统计目录: {temp_dir}")
                except Exception as e3:
                    logger.error(f"创建临时统计目录时出错: {e3}")
                    # 无法创建目录，使用内存模式
                    self.stats_dir = None
                    logger.warning(
                        "无法创建统计目录，将使用内存模式，不会保存统计文件")

    def load_stats(self, days=7, cache_name=None):
        """
        加载缓存统计数据

        Args:
            days: 加载最近几天的数据，0表示所有数据
            cache_name: 指定缓存名称，None表示所有缓存

        Returns:
            dict: 加载的统计数据
        """
        # 清空现有数据
        self.cache_stats = {}

        # 如果统计目录无效，返回空结果
        if self.stats_dir is None:
            logger.warning("统计目录无效，无法加载统计数据")
            return {}

        # 计算时间范围
        cutoff_time = None
        if days > 0:
            cutoff_time = time.time() - days * 86400

        try:
            # 确定文件模式
            if cache_name:
                pattern = os.path.join(self.stats_dir,
                                       f"{cache_name}_stats*.json")
            else:
                pattern = os.path.join(self.stats_dir, "*_stats*.json")

            # 查找匹配的文件
            stat_files = glob.glob(pattern)
            total_files = 0

            if not stat_files:
                logger.warning(f"在 {self.stats_dir} 中未找到任何统计文件")
                return {}

            # 处理每个文件
            for stat_file in stat_files:
                try:
                    # 确保文件存在并可读
                    if not os.path.exists(stat_file) or not os.path.isfile(
                            stat_file):
                        logger.warning(f"统计文件不存在或不是文件: {stat_file}")
                        continue

                    # 检查文件修改时间
                    file_mtime = os.path.getmtime(stat_file)
                    if cutoff_time and file_mtime < cutoff_time:
                        continue

                    # 解析文件名获取缓存名称
                    file_name = os.path.basename(stat_file)
                    cache_name_from_file = file_name.split('_stats')[0]

                    # 加载统计数据
                    with open(stat_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)

                    # 确保缓存名称存在
                    if 'name' not in stats:
                        stats['name'] = cache_name_from_file

                    # 添加文件元数据
                    stats['_file'] = stat_file
                    stats['_mtime'] = file_mtime

                    # 添加到数据集
                    cache_key = stats['name']
                    if cache_key not in self.cache_stats:
                        self.cache_stats[cache_key] = []

                    self.cache_stats[cache_key].append(stats)
                    total_files += 1

                except Exception as e:
                    logger.error(f"处理统计文件 {stat_file} 时出错: {e}")
                    logger.debug(traceback.format_exc())

            # 对每个缓存的统计数据按时间排序
            for name in self.cache_stats:
                self.cache_stats[name].sort(key=lambda x: x.get('_mtime', 0))

                # 提取时间序列数据
                self._extract_time_series(name)

            logger.info(
                f"已加载 {total_files} 个统计文件，涉及 {len(self.cache_stats)} 个缓存")

        except Exception as e:
            logger.error(f"加载统计数据时出错: {e}")
            logger.debug(traceback.format_exc())

        return self.cache_stats

    def _extract_time_series(self, cache_name):
        """
        从统计数据中提取时间序列

        Args:
            cache_name: 缓存名称
        """
        if cache_name not in self.cache_stats:
            return

        stats_list = self.cache_stats[cache_name]
        series = {
            'timestamps': [],
            'hit_rates': [],
            'usage_rates': [],
            'sizes': []
        }

        for stats in stats_list:
            try:
                # 获取时间戳
                timestamp = stats.get('_mtime', 0)

                # 提取命中率
                hit_rate = parse_percentage(stats.get('hit_rate', '0%'))

                # 提取使用率
                usage = parse_percentage(stats.get('usage_percent', '0%'))

                # 提取大小
                size = float(stats.get('size', 0))

                # 添加到系列
                series['timestamps'].append(timestamp)
                series['hit_rates'].append(hit_rate)
                series['usage_rates'].append(usage)
                series['sizes'].append(size)
            except Exception as e:
                logger.error(f"提取时间序列数据时出错: {e}")

        self.time_series[cache_name] = series

    def analyze_trends(self, cache_name=None):
        """
        分析缓存性能趋势

        Args:
            cache_name: 指定缓存名称，None表示所有缓存

        Returns:
            dict: 趋势分析结果
        """
        trends = {}

        # 确定要分析的缓存
        names_to_analyze = [
            cache_name] if cache_name else self.time_series.keys()

        for name in names_to_analyze:
            if name not in self.time_series:
                continue

            series = self.time_series[name]

            # 确保有足够的数据点
            if len(series['timestamps']) < 2:
                continue

            try:
                # 计算趋势
                first_idx = 0
                last_idx = len(series['timestamps']) - 1

                # 计算时间跨度
                time_span = series['timestamps'][last_idx] - \
                            series['timestamps'][first_idx]
                if time_span <= 0:
                    continue

                # 计算每日变化率
                daily_change = 86400 / time_span  # 转换为每天

                # 计算各指标的变化
                hit_rate_change = (series['hit_rates'][last_idx] -
                                   series['hit_rates'][
                                       first_idx]) * daily_change
                usage_change = (series['usage_rates'][last_idx] -
                                series['usage_rates'][first_idx]) * daily_change
                size_change = (series['sizes'][last_idx] - series['sizes'][
                    first_idx]) * daily_change

                # 确定趋势方向
                hit_rate_direction = get_trend_direction(hit_rate_change)
                usage_direction = get_trend_direction(usage_change)
                size_direction = get_trend_direction(size_change, 1)

                # 记录趋势
                trends[name] = {
                    'time_span_days': time_span / 86400,
                    'hit_rate': {
                        'start': series['hit_rates'][first_idx],
                        'end': series['hit_rates'][last_idx],
                        'daily_change': hit_rate_change,
                        'direction': hit_rate_direction
                    },
                    'usage': {
                        'start': series['usage_rates'][first_idx],
                        'end': series['usage_rates'][last_idx],
                        'daily_change': usage_change,
                        'direction': usage_direction
                    },
                    'size': {
                        'start': series['sizes'][first_idx],
                        'end': series['sizes'][last_idx],
                        'daily_change': size_change,
                        'direction': size_direction
                    }
                }
            except Exception as e:
                logger.error(f"分析缓存 {name} 趋势时出错: {e}")
                logger.debug(traceback.format_exc())

        return trends

    def get_recommendations(self, cache_name=None):
        """
        生成缓存优化建议

        Args:
            cache_name: 指定缓存名称，None表示所有缓存

        Returns:
            dict: 按缓存名称组织的建议列表
        """
        recommendations = {}

        # 获取趋势数据
        trends = self.analyze_trends(cache_name)

        # 确定要分析的缓存
        names_to_analyze = [
            cache_name] if cache_name else self.cache_stats.keys()

        for name in names_to_analyze:
            if name not in self.cache_stats or not self.cache_stats[name]:
                continue

            try:
                # 获取最新统计数据
                latest = self.cache_stats[name][-1]
                cache_recs = []

                # 提取关键指标
                hit_rate = parse_percentage(latest.get('hit_rate', '0%'))
                usage = parse_percentage(latest.get('usage_percent', '0%'))
                size = float(latest.get('size', 0))
                capacity = float(latest.get('capacity', 0))

                # 1. 命中率相关建议
                if hit_rate < 0.4 and hit_rate > 0:
                    if name in trends and trends[name]['hit_rate'][
                        'direction'] == "下降":
                        cache_recs.append(
                            "命中率低且呈下降趋势，建议检查缓存键生成或调整缓存项生命周期")
                    else:
                        cache_recs.append(
                            "命中率低，考虑优化缓存键生成或缓存策略")

                # 2. 使用率相关建议
                if usage < 0.3 and capacity > 20:
                    optimal_capacity = max(int(size * 1.5), 10)
                    cache_recs.append(
                        f"缓存使用率低({latest.get('usage_percent', '0%')})，建议将容量从{capacity}减少到{optimal_capacity}")
                elif usage > 0.9:
                    optimal_capacity = int(capacity * 1.3)
                    cache_recs.append(
                        f"缓存接近容量上限({latest.get('usage_percent', '0%')})，考虑增加容量至{optimal_capacity}")

                # 3. 趋势相关建议
                if name in trends:
                    trend = trends[name]

                    # 命中率快速下降
                    if trend['hit_rate']['daily_change'] < -0.05:  # 每天下降超过5%
                        cache_recs.append(
                            f"命中率快速下降(每天{trend['hit_rate']['daily_change'] * 100:.1f}%)，检查请求模式变化")

                    # 使用率快速上升
                    if trend['usage']['daily_change'] > 0.1:  # 每天上升超过10%
                        days_until_full = (1 - trend['usage']['end']) / \
                                          trend['usage']['daily_change'] if \
                        trend['usage']['daily_change'] > 0 else float('inf')
                        if days_until_full < 7:  # 一周内将达到容量上限
                            cache_recs.append(
                                f"使用率快速上升，预计{days_until_full:.1f}天后达到容量上限，建议提前扩容")

                recommendations[name] = cache_recs
            except Exception as e:
                logger.error(f"为缓存 {name} 生成建议时出错: {e}")
                logger.debug(traceback.format_exc())
                recommendations[name] = []

        return recommendations

    def create_report(self, output_file=None):
        """
        创建缓存分析报告

        Args:
            output_file: 输出文件路径，None则使用默认路径

        Returns:
            str: 报告文件路径
        """
        # 如果统计目录无效且未指定输出文件，则无法生成报告
        if self.stats_dir is None and output_file is None:
            logger.error("统计目录无效且未指定输出文件路径，无法生成报告")
            return None

        # 如果未指定输出文件，使用默认路径
        if not output_file:
            if self.stats_dir:
                # 在统计目录中创建报告
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.stats_dir,
                                           f"cache_analysis_{timestamp}.txt")
            else:
                # 如果统计目录无效，尝试在当前目录或临时目录创建
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"cache_analysis_{timestamp}.txt"
                except Exception:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(tempfile.gettempdir(),
                                               f"cache_analysis_{timestamp}.txt")

        # 获取趋势和建议
        trends = self.analyze_trends()
        recommendations = self.get_recommendations()

        try:
            # 确保输出文件所在的目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"创建输出目录时出错: {e}")
                    # 尝试使用临时目录
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(tempfile.gettempdir(),
                                               f"cache_analysis_{timestamp}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入报告标题
                f.write("=" * 70 + "\n")
                f.write(
                    f"缓存系统分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")

                # 写入总体信息
                f.write(f"分析的缓存数量: {len(self.cache_stats)}\n")
                f.write(f"数据时间范围: {days_range(self.cache_stats)}\n\n")

                # 写入每个缓存的详细分析
                for name in sorted(self.cache_stats.keys()):
                    if not self.cache_stats[name]:
                        continue

                    f.write("-" * 70 + "\n")
                    f.write(f"缓存: {name}\n")
                    f.write("-" * 70 + "\n\n")

                    # 获取最新统计
                    latest = self.cache_stats[name][-1]

                    # 写入基本信息
                    f.write("● 基本信息:\n")
                    for key in ['size', 'capacity', 'hit_rate',
                                'usage_percent']:
                        if key in latest:
                            f.write(f"  {key}: {latest[key]}\n")

                    # 写入趋势分析
                    if name in trends:
                        trend = trends[name]
                        f.write("\n● 趋势分析:\n")
                        f.write(
                            f"  时间跨度: {trend['time_span_days']:.1f}天\n")

                        hit_rate = trend['hit_rate']
                        f.write(
                            f"  命中率: {hit_rate['start'] * 100:.1f}% → {hit_rate['end'] * 100:.1f}%")
                        f.write(
                            f" ({hit_rate['direction']}, 每天{hit_rate['daily_change'] * 100:+.1f}%)\n")

                        usage = trend['usage']
                        f.write(
                            f"  使用率: {usage['start'] * 100:.1f}% → {usage['end'] * 100:.1f}%")
                        f.write(
                            f" ({usage['direction']}, 每天{usage['daily_change'] * 100:+.1f}%)\n")

                        size = trend['size']
                        f.write(
                            f"  缓存大小: {size['start']} → {size['end']}项")
                        f.write(
                            f" ({size['direction']}, 每天{size['daily_change']:+.1f}项)\n")

                    # 写入优化建议
                    if name in recommendations and recommendations[name]:
                        f.write("\n● 优化建议:\n")
                        for i, rec in enumerate(recommendations[name], 1):
                            f.write(f"  {i}. {rec}\n")

                    f.write("\n")

                # 写入系统级建议
                all_recs = []
                for name, recs in recommendations.items():
                    for rec in recs:
                        all_recs.append((name, rec))

                if all_recs:
                    f.write("=" * 70 + "\n")
                    f.write("系统级优化建议\n")
                    f.write("=" * 70 + "\n\n")

                    # 分类整理建议
                    memory_recs = [(name, rec) for name, rec in all_recs if
                                   "容量" in rec]
                    performance_recs = [(name, rec) for name, rec in all_recs if
                                        "命中率" in rec or "性能" in rec]
                    other_recs = [(name, rec) for name, rec in all_recs if
                                  (name, rec) not in memory_recs and (
                                  name, rec) not in performance_recs]

                    if memory_recs:
                        f.write("内存优化:\n")
                        for i, (name, rec) in enumerate(memory_recs, 1):
                            f.write(f"{i}. [{name}] {rec}\n")
                        f.write("\n")

                    if performance_recs:
                        f.write("性能优化:\n")
                        for i, (name, rec) in enumerate(performance_recs, 1):
                            f.write(f"{i}. [{name}] {rec}\n")
                        f.write("\n")

                    if other_recs:
                        f.write("其他建议:\n")
                        for i, (name, rec) in enumerate(other_recs, 1):
                            f.write(f"{i}. [{name}] {rec}\n")
                        f.write("\n")

                # 写入结束信息
                f.write("=" * 70 + "\n")
                f.write(
                    f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")

            logger.info(f"已创建缓存分析报告: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"创建缓存分析报告时出错: {e}")
            logger.debug(traceback.format_exc())
            return None


def days_range(cache_stats):
    """
    计算统计数据的时间跨度

    Args:
        cache_stats: 缓存统计数据字典

    Returns:
        str: 格式化的时间跨度字符串
    """
    all_timestamps = []

    for stats_list in cache_stats.values():
        for stats in stats_list:
            if '_mtime' in stats:
                all_timestamps.append(stats['_mtime'])

    if not all_timestamps:
        return "未知"

    earliest = min(all_timestamps)
    latest = max(all_timestamps)
    days = (latest - earliest) / 86400

    if days < 1:
        return f"{days * 24:.1f}小时"
    else:
        return f"{days:.1f}天"


def parse_percentage(value):
    """
    解析百分比字符串或数值，返回0-1之间的浮点数

    Args:
        value: 输入值，可以是字符串(如"75%")或数值

    Returns:
        float: 0-1之间的浮点数
    """
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            if value.endswith('%'):
                return float(value.rstrip('%')) / 100.0
            else:
                return float(value)
        return 0.0
    except (TypeError, ValueError):
        return 0.0


def get_trend_direction(change_value, threshold=0.01):
    """
    根据变化值确定趋势方向

    Args:
        change_value: 变化值
        threshold: 变化阈值，默认0.01

    Returns:
        str: 趋势方向("上升"、"下降"或"稳定")
    """
    if change_value > threshold:
        return "上升"
    elif change_value < -threshold:
        return "下降"
    else:
        return "稳定"


def analyze_caches(days=7, output_file=None, stats_dir=None):
    """
    分析缓存并创建报告

    Args:
        days: 分析最近几天的数据，0表示所有数据
        output_file: 输出文件路径，None则使用默认路径
        stats_dir: 统计数据目录，None则使用默认目录

    Returns:
        str: 报告文件路径或None（如果分析失败）
    """
    try:
        # 确保日志目录存在
        try:
            log_dir = "logs"
            cache_stats_dir = "logs/cache_stats"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            if not os.path.exists(cache_stats_dir):
                os.makedirs(cache_stats_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"创建日志目录时出错: {e}")

        # 使用默认的统计目录，如果未指定
        if stats_dir is None:
            stats_dir = "logs/cache_stats"

        # 确保stats_dir是一个有效的字符串路径
        if not isinstance(stats_dir, str):
            logger.error(f"无效的统计目录类型: {type(stats_dir)}")
            stats_dir = "logs/cache_stats"  # 使用默认路径

        # 确保目录存在
        if not os.path.exists(stats_dir):
            try:
                os.makedirs(stats_dir, exist_ok=True)
                logger.info(f"已创建缓存统计目录: {stats_dir}")
            except Exception as e:
                logger.error(f"创建统计目录时出错: {e}")
                # 使用临时目录
                stats_dir = os.path.join(tempfile.gettempdir(), "cache_stats")
                os.makedirs(stats_dir, exist_ok=True)

        # 创建分析器
        analyzer = CacheAnalyzer(stats_dir=stats_dir)

        # 加载统计数据
        loaded_stats = analyzer.load_stats(days=days)

        if not loaded_stats or not any(loaded_stats.values()):
            logger.warning("未找到缓存统计数据进行分析")
            print("未找到缓存统计数据，请确保缓存系统已运行并生成统计信息")
            return None

        # 创建报告
        report_path = analyzer.create_report(output_file)

        if report_path:
            logger.info(f"成功创建缓存分析报告: {report_path}")
            print(f"分析报告已生成: {report_path}")

            # 显示报告预览
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    preview = f.read(800)
                    print("\n报告预览:")
                    print("-" * 40)
                    print(preview + ("..." if len(preview) >= 800 else ""))
                    print("-" * 40)
            except Exception as e:
                logger.error(f"显示报告预览时出错: {e}")

            return report_path
        else:
            logger.error("创建缓存分析报告失败")
            print("创建分析报告失败，请查看日志获取详细信息")
            return None

    except Exception as e:
        logger.error(f"分析缓存时出错: {e}")
        logger.debug(traceback.format_exc())
        print(f"分析缓存时出错: {e}")
        return None


def get_summary():
    """
    获取缓存系统摘要信息

    Returns:
        dict: 缓存系统摘要信息字典
    """
    try:
        # 尝试导入缓存监控器
        try:
            from .cache_monitor import get_monitor
        except ImportError:
            # 尝试直接导入
            try:
                from cache_monitor import get_monitor
            except ImportError:
                return {
                    'cache_count': 0,
                    'hit_rate': 0,
                    'usage': 0,
                    'issues': ["无法导入缓存监控器"]
                }

        # 获取监控器
        monitor = get_monitor()

        # 确保监控已启动
        if hasattr(monitor, 'monitoring') and not monitor.monitoring:
            monitor.start_monitoring()

        # 获取所有缓存状态
        current_stats = monitor.get_all_stats()

        # 统计缓存数
        cache_count = len(current_stats)

        # 如果没有缓存，返回空摘要
        if cache_count == 0:
            return {
                'cache_count': 0,
                'hit_rate': 0,
                'usage': 0,
                'issues': ["没有注册的缓存实例"]
            }

        # 统计总体命中率
        total_hits = sum(
            int(stats.get('hit_count', 0)) for stats in current_stats.values())
        total_misses = sum(
            int(stats.get('miss_count', 0)) for stats in current_stats.values())
        overall_hit_rate = total_hits / (total_hits + total_misses) * 100 if (
                                                                                         total_hits + total_misses) > 0 else 0

        # 统计总体使用率
        total_size = sum(
            int(stats.get('size', 0)) for stats in current_stats.values())
        total_capacity = sum(
            int(stats.get('capacity', 0)) for stats in current_stats.values())
        overall_usage = total_size / total_capacity * 100 if total_capacity > 0 else 0

        # 统计健康状态
        issues = []
        for name, stats in current_stats.items():
            # 解析命中率
            hit_rate = parse_percentage(stats.get('hit_rate', '0%')) * 100

            # 解析使用率
            usage = parse_percentage(stats.get('usage_percent', '0%')) * 100

            # 检查问题
            if hit_rate < 40 and hit_rate > 0:
                issues.append(f"缓存'{name}'命中率低({hit_rate:.1f}%)")

            if usage > 90:
                issues.append(f"缓存'{name}'接近容量上限({usage:.1f}%)")
            elif usage < 20 and int(stats.get('capacity', 0)) > 10:
                issues.append(f"缓存'{name}'使用率过低({usage:.1f}%)")

        return {
            'cache_count': cache_count,
            'hit_rate': overall_hit_rate,
            'usage': overall_usage,
            'issues': issues
        }
    except Exception as e:
        logger.error(f"获取缓存摘要时出错: {e}")
        logger.debug(traceback.format_exc())
        return {
            'cache_count': 0,
            'hit_rate': 0,
            'usage': 0,
            'issues': [f"获取缓存摘要时出错: {e}"]
        }


if __name__ == "__main__":
    # 简单的命令行使用示例
    import argparse

    # 配置命令行参数
    parser = argparse.ArgumentParser(description="缓存分析工具")
    parser.add_argument("--days", type=int, default=7,
                        help="分析最近几天的数据，0表示所有数据")
    parser.add_argument("--output", type=str, help="输出报告文件路径")
    parser.add_argument("--dir", type=str, help="缓存统计目录路径")
    parser.add_argument("--stats", action="store_true",
                        help="只显示摘要统计信息，不生成报告")

    args = parser.parse_args()

    print("=" * 50)
    print("缓存分析工具")
    print("=" * 50)

    if args.stats:
        # 显示摘要统计
        try:
            summary = get_summary()
            print(f"缓存数量: {summary['cache_count']}")
            print(f"总体命中率: {summary['hit_rate']:.1f}%")
            print(f"总体使用率: {summary['usage']:.1f}%")

            if summary['issues']:
                print("\n检测到的问题:")
                for issue in summary['issues']:
                    print(f"- {issue}")
            else:
                print("\n未检测到问题")
        except Exception as e:
            print(f"获取摘要统计时出错: {e}")
    else:
        # 生成分析报告
        print("正在分析缓存数据...")
        try:
            report_file = analyze_caches(days=args.days,
                                         output_file=args.output,
                                         stats_dir=args.dir)

            if report_file:
                print(f"\n分析完成，报告已保存至: {report_file}")
            else:
                print("\n分析失败，无法生成报告")
        except Exception as e:
            print(f"生成报告时出错: {e}")
            import traceback

            print(traceback.format_exc())
