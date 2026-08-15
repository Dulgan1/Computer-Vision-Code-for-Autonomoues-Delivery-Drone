import cv2
import config


class Camera:
    def __init__(
            self, index=config.CAMERA_INDEX, 
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            fps=config.FPS
            ):

        self.cap = cv2.VideoCapture(index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Supported by V4L2 cameras; ignored harmlessly by backends that do
        # not expose capture-buffer control. It reduces stale-frame latency.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)

    def read(self):
        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self):
        self.cap.release()
