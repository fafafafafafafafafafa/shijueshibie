"""
线程池管理工具 - utils包的一部分

提供全局线程池管理功能，用于：
- 获取应用范围内共享的线程池
- 安全地关闭线程池
"""
import concurrent.futures

# 全局线程池
_thread_pool = None

def get_thread_pool(max_workers=None):
    """获取或创建线程池"""
    global _thread_pool
    if _thread_pool is None:
        max_workers = max_workers or 2  # 默认2个工作线程
        _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool

def shutdown_thread_pool():
    """关闭线程池"""
    global _thread_pool
    if _thread_pool:
        _thread_pool.shutdown()
        _thread_pool = None
