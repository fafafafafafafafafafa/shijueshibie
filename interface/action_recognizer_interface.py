from abc import ABC, abstractmethod


class ActionRecognizerInterface(ABC):
    """
    动作识别器接口，定义动作识别器必须实现的方法
    """

    @abstractmethod
    def recognize_action(self, person):
        """
        识别人体动作

        Args:
            person: 包含人体信息的字典，包括关键点、位置等

        Returns:
            str: 识别的动作名称
        """
        pass

    @abstractmethod
    def release_resources(self):
        """
        释放识别器使用的资源

        Returns:
            bool: 资源释放是否成功
        """
        pass

    @abstractmethod
    def toggle_feature(self, feature_name, state):
        """
        切换识别器特定功能

        Args:
            feature_name: 功能名称 (例如 'ml_model', 'dtw', 'threading')
            state: 要设置的状态 (True/False)

        Returns:
            bool: 是否成功切换功能
        """
        pass
