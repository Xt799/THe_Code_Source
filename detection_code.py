import cv2
import numpy as np
from pypylon import pylon
from ultralytics import YOLO

import rclpy
from rclpy.node import Node


class BaslerYoloNode(Node):
    def __init__(self):
        super().__init__('basler_yolo_node')

        # Load YOLOv8 pose model
        self.model = YOLO('./src/basler_yolo_node/robotv6.pt')

        # Set up Basler camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # Image converter
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        self.get_logger().info("Basler YOLO Node started.")

    def run(self):
        while rclpy.ok() and self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Convert image to OpenCV format
                image = self.converter.Convert(grabResult).GetArray()
                canvas = image.copy()

                # Run YOLO pose detection
                results = self.model(image)

                for result in results:
                    keypoints = result.keypoints.xy
                    if keypoints is None or len(keypoints) == 0:
                        continue

                    for robot in keypoints:
                        robot = robot.cpu().numpy()

                        if len(robot) < 4:
                            continue  # Skip if fewer than 4 keypoints

                        points = np.array([
                            robot[0],  # FL
                            robot[1],  # FR
                            robot[2],  # BR
                            robot[3],  # BL
                        ], dtype=np.float32)

                        points_int = points.astype(int)

                        # Draw bounding lines
                        for i in range(4):
                            cv2.line(canvas, tuple(points_int[i]),
                                     tuple(points_int[(i + 1) % 4]), (100, 255, 100), 2)

                        # Draw keypoints
                        point_names = ['FL', 'FR', 'BR', 'BL']
                        point_colors = [(0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 0, 255)]
                        for i, pt in enumerate(points_int):
                            cv2.circle(canvas, tuple(pt), 6, point_colors[i], -1)
                            cv2.putText(canvas, point_names[i], tuple(pt + [6, -6]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, point_colors[i], 2)

                # Show image
                cv2.imshow("YOLO Pose Detection", canvas)

            if cv2.waitKey(1) == 27:  # ESC to exit
                break

        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()
        self.camera.StopGrabbing()
        self.camera.Close()
        self.get_logger().info("Basler YOLO Node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = BaslerYoloNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
