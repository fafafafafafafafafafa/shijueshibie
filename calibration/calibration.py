import cv2
import time
import math
import logging


class CalibrationManager:
    """负责处理系统校准过程，更加模块化和健壮"""

    def __init__(self, detector, position_mapper,event_system=None):
        self.detector = detector
        self.position_mapper = position_mapper
        self.calibration_height = None
        self.logger = logging.getLogger("CalibrationManager")
        self.events = event_system  # 添加事件系统引用

    def calibrate(self, camera, duration=3):
        """执行校准过程，获取人体参考高度

        Args:
            camera: OpenCV视频捕获对象
            duration: 校准持续时间（秒）

        Returns:
            bool: 校准是否成功
        """
        self.logger.info("=== 校准开始 ===")
        print("\n=== 校准开始 ===")
        print("请站在距离相机大约1米的位置，双臂自然下垂")

        heights = []
        start_time = cv2.getTickCount()
        last_second = 0

        try:
            # 创建校准窗口并调整大小
            cv2.namedWindow('校准', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('校准', 800, 600)

            while (
                    cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
                ret, frame = camera.read()
                if not ret:
                    self.logger.warning("相机读取错误，校准期间")
                    print("相机读取错误，校准期间")
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)

                # 显示倒计时
                elapsed = (
                                      cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                current_second = int(elapsed)
                remaining = duration - elapsed

                if current_second != last_second:
                    print(f"校准倒计时: {int(remaining) + 1}秒...")
                    last_second = current_second

                calibration_text = f"校准中... {int(remaining) + 1}"

                try:
                    persons = self.detector.detect_pose(frame)
                    if persons:
                        person = persons[0]
                        heights.append(person['height'])

                        # 在画面中绘制人体骨架
                        self.detector.draw_skeleton(frame, person['keypoints'])

                        # 显示当前高度
                        height_text = f"高度: {person['height']:.1f}"
                        cv2.putText(frame, height_text,
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

                        # 显示检测到的人数
                        cv2.putText(frame, f"检测到 1 人",
                                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "未检测到人体! 请站在相机前方",
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                except Exception as e:
                    self.logger.error(f"校准期间检测错误: {e}")
                    print(f"校准期间检测错误: {e}")

                # 显示校准状态
                cv2.putText(frame, calibration_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('校准', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 计算平均值作为校准基准
            if heights:
                # 过滤掉异常值 (超出平均值1个标准差的值)
                mean_height = sum(heights) / len(heights)
                std_dev = math.sqrt(
                    sum((h - mean_height) ** 2 for h in heights) / len(heights))
                valid_heights = [h for h in heights if
                                 abs(h - mean_height) <= std_dev]

                calibration_success = False
                if valid_heights:
                    # 使用中位数可能更稳定
                    valid_heights.sort()
                    self.calibration_height = valid_heights[
                        len(valid_heights) // 2]
                    # 使用90%的高度作为参考，增大深度可视范围
                    self.position_mapper.set_calibration(
                        self.calibration_height)
                    self.logger.info(
                        f"校准完成. 参考高度: {self.calibration_height:.1f}")
                    print(f"校准完成. 参考高度: {self.calibration_height:.1f}")
                    calibration_success = True

                    # 发布校准成功事件
                    if hasattr(self, 'events') and self.events:
                        self.events.publish("calibration_completed", {
                            'height': self.calibration_height,
                            'success': True,
                            'timestamp': time.time(),
                            'valid_samples': len(valid_heights),
                            'total_samples': len(heights)
                        })

                    return True

            # 校准失败
            self.logger.warning("校准失败. 未获取有效高度数据. 使用默认值.")
            print("校准失败. 未检测到人体. 使用默认值.")
            self.position_mapper.set_calibration(200)  # 使用默认值

            # 发布校准失败事件
            if hasattr(self, 'events') and self.events:
                self.events.publish("calibration_failed", {
                    'reason': "未获取有效高度数据",
                    'timestamp': time.time(),
                    'default_height': 200
                })

            return False

        except Exception as e:
            self.logger.error(f"校准过程失败: {e}")
            print(f"校准过程失败: {e}")
            print("使用默认校准值")
            self.position_mapper.set_calibration(200)

            # 发布校准错误事件
            if hasattr(self, 'events') and self.events:
                self.events.publish("calibration_error", {
                    'error': str(e),
                    'timestamp': time.time(),
                    'default_height': 200
                })

            return False

        finally:
            cv2.destroyWindow('校准')






