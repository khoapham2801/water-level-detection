import cv2
from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage as PILPhotoImage


class VideoReader:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError('Unable to open video source', video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        self.release()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag
                # return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return ret, frame

        return None, None

    def release(self):
        if self.vid.isOpened():
            self.vid.release()
