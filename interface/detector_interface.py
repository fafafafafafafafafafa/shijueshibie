from abc import ABC, abstractmethod


class PersonDetectorInterface(ABC):
    """
    人体检测器接口，定义检测器必须实现的方法
    """

    @abstractmethod
    def detect_pose(self, frame):
        """
        检测输入帧中的人体姿态

        Args:
            frame: 输入的图像帧，通常是BGR格式的numpy数组

        Returns:
            list: 检测到的人体列表，每个人体包含关键点、边界框等信息
        """
        pass

    @abstractmethod
    def release_resources(self):
        """
        释放检测器使用的资源

        Returns:
            bool: 资源释放是否成功
        """
        pass

    @abstractmethod
    def draw_skeleton(self, frame, keypoints):
        """
        在图像上绘制骨架

        Args:
            frame: 输入的图像帧
            keypoints: 关键点列表

        Returns:
            ndarray: 带有骨架的图像
        """
        pass

    @abstractmethod
    def toggle_feature(self, feature_name, state):
        """
        切换检测器特定功能

        Args:
            feature_name: 功能名称 (例如 'mediapipe')
            state: 要设置的状态 (True/False)

        Returns:
            bool: 是否成功切换功能
        """
        pass

    @abstractmethod
    def get_feature_state(self, feature_name):
        """
        获取检测器特定功能的状态

        Args:
            feature_name: 功能名称 (例如 'mediapipe')

        Returns:
            bool: 功能当前状态
        """
        pass
