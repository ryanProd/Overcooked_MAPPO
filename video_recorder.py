import cv2
import imageio
import os

class VideoRecorder(object):
    def __init__(self, save_dir, height=400, width=400, fps=7):
        self.save_dir = save_dir
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def record(self, env):
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frames.append(frame)

    def save(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        imageio.mimwrite(path, self.frames)