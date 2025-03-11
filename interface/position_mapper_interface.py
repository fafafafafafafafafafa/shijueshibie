from abc import ABC, abstractmethod


class PositionMapperInterface(ABC):
    """
    位置映射器接口，定义位置映射器必须实现的方法
    """

    @abstractmethod
    def map_position_to_room(self, frame_width, frame_height, room_width,
                             room_height, person):
        """
        将摄像头坐标映射到房间平面图坐标

        Args:
            frame_width: 帧宽度
            frame_height: 帧高度
            room_width: 房间宽度
            room_height: 房间高度
            person: 包含人体信息的字典

        Returns:
            tuple: (room_x, room_y, depth) 房间坐标和深度
        """
        pass

    @abstractmethod
    def set_calibration(self, height):
        """
        设置校准高度

        Args:
            height: 校准高度值
        """
        pass

    @abstractmethod
    def get_stable_position(self, room_x, room_y, action):
        """
        获取稳定的位置

        Args:
            room_x: 房间X坐标
            room_y: 房间Y坐标
            action: 当前动作

        Returns:
            tuple: (stable_x, stable_y) 稳定的位置坐标
        """
        pass

    @abstractmethod
    def smooth_position(self, x, y):
        """
        平滑位置数据

        Args:
            x: X坐标
            y: Y坐标

        Returns:
            tuple: (smooth_x, smooth_y) 平滑后的位置坐标
        """
        pass

    @abstractmethod
    def release_resources(self):
        """
        释放映射器使用的资源

        Returns:
            bool: 资源释放是否成功
        """
        pass

    @abstractmethod
    def toggle_feature(self, feature_name, state):
        """
        切换映射器特定功能

        Args:
            feature_name: 功能名称
            state: 要设置的状态 (True/False)

        Returns:
            bool: 是否成功切换功能
        """
        pass
