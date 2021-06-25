from os import system, write
import tkinter as tk
import cv2
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showerror
import pygubu
from pygubu.builder import ttkstdwidgets
from videoReader import VideoReader
from pattern import EPattern
from waterLevel import WaterLevel
from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage as PILPhotoImage


class Application:
    def __init__(self, root):
        self.root = root
        self.videoReader = None
        # self.pattern = None
        self.image = None
        self.job = None
        self.pause = False
        self.fileName = None
        self.root.title('Đo mực nước')
        self.builder = pygubu.Builder()
        self.builder.add_from_file('ui/test_main.ui')
        self.rulerFootHeight = None
        self.rulerHeight = None
        self.threeLinesHeight = None
        self.mainFrame = self.builder.get_object('mainFrame', self.root)
        self.waterLevel = WaterLevel()
        self.labelRulerFootHeight = self.builder.get_object(
            'labelRulerFootHeight', self.root)
        self.labelRulerHeight = self.builder.get_object(
            'labelRulerHeight', self.root)
        self.labelThreeLinesHeight = self.builder.get_object(
            'labelThreeLinesHeight', self.root)
        self.labelWaterLevel = self.builder.get_object(
            'labelRulerFootHeight', self.root)
        self.txtRulerFootHeight = self.builder.get_object(
            'txtRulerFootHeight', self.root)
        self.txtRulerHeight = self.builder.get_object(
            'txtRulerHeight', self.root)
        self.txtThreeLinesHeight = self.builder.get_object(
            'txtThreeLinesHeight', self.root)

        self.btnOpenFile = self.builder.get_object(
            'btnOpenFile', self.root)
        self.btnOpenFile.bind('<ButtonPress-1>', self.open_file)

        self.btnDetectWaterLevel = self.builder.get_object(
            'btnDetectWaterLevel', self.root)
        self.btnDetectWaterLevel.bind(
            '<ButtonPress-1>', self.detect_water_level)

        self.btnQuit = self.builder.get_object('btnQuit', self.root)
        self.btnQuit.bind('<ButtonPress-1>', self.exit)

        self.canvas = self.builder.get_object('canvas', self.root)
        self.canvas.bind('<ButtonRelease-1>', self.click_canvas)

        self.lblWaterLevel = self.builder.get_object(
            'lblWaterLevel', self.root)
        # self.mainFrame.grid(fill=BOTH, expand=1)
        self.mainFrame.pack(fill=BOTH, expand=1)
        self.builder.connect_callbacks(self)

    def detect_water_level(self, event):
        try:
            self.rulerFootHeight = float(self.txtRulerFootHeight.get())
        except ValueError:
            showerror(title="Xảy ra lỗi",
                      message="Độ cao chân thước không hợp lệ!")
            return

        try:
            self.rulerHeight = float(self.txtRulerHeight.get())
        except ValueError:
            showerror(title="Xảy ra lỗi",
                      message="Độ dài thước không hợp lệ!")
            return

        try:
            self.threeLinesHeight = float(self.txtThreeLinesHeight.get())
        except ValueError:
            showerror(title="Xảy ra lỗi",
                      message="Độ dài 3 vạch không hợp lệ!")
            return
        if self.fileName is None:
            showerror(title="Xảy ra lỗi",
                      message="Chưa có ảnh/video!")
        fileName = self.fileName
        if fileName:
            if ".jpg" in fileName or ".png" in fileName:
                self.image = cv2.imread(fileName)
                self.display_image()
            elif ".avi" in fileName or ".mp4" in fileName:
                self.stop_video()
                self.videoReader = VideoReader(fileName)
                self.play_video()

    def open_file(self, event):
        fileName = filedialog.askopenfilename(title='Chọn ảnh/video', filetypes=(
            ('jpg files', '*.jpg'), ('mp4 files', '*.mp4'), ('png files', '*.png'),
            ('avi files', '*.avi')
        ))
        self.fileName = fileName
        return self.fileName

    def display_image(self):
        img, waterLevel, rulerHeight = self.waterLevel.calculateWaterLevel(
            self.image, self.rulerFootHeight, self.rulerHeight, self.threeLinesHeight)
        self.lblWaterLevel.config(text=str(waterLevel))
        img_result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height = img_result.shape[1], img_result.shape[0]
        self.canvas.config(width=width, height=height)
        self.image = PILPhotoImage(image=fromarray(img_result))
        self.canvas.create_image(0, 0, image=self.image, anchor=NW)

    def play_video(self, delay=50):
        # Get a frame from the video source
        ret, frame = self.videoReader.get_frame()
        frame, waterLevel, rulerHeight = self.waterLevel.calculateWaterLevel(
            frame, self.rulerFootHeight, self.rulerHeight, self.threeLinesHeight)
        self.lblWaterLevel.config(text=str(waterLevel))
        frame_result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height = self.videoReader.width, self.videoReader.height
        self.canvas.config(width=width, height=height)
        self.image = PILPhotoImage(image=fromarray(frame_result))
        self.canvas.create_image(0, 0, image=self.image, anchor=NW)
        # print(frame.size())

        self.job = self.root.after(delay, self.play_video)

    def stop_video(self):
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.videoReader.release()
            self.job = None

    def pause_video(self):
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job = None

    def resume_video(self):
        if self.videoReader:
            self.play_video()

    def click_canvas(self, event):
        if self.pause:
            self.resume_video()
        else:
            self.pause_video()
        self.pause = not self.pause

    def exit(self, event):
        self.root.quit()


if __name__ == '__main__':
    root = tk.Tk()
    root.state("zoomed")
    app = Application(root)
    f = open("count.txt", "r")
    i = f.read()
    j = int(i)
    f.close()
    if j > 10:
        showerror(title="Xảy ra lỗi",
                  message="Bạn đã sử dụng quá số lần demo!")
        root.destroy()
    f = open("count.txt", "r+")
    j += 1
    f.write(str(j))
    f.close()
    root.mainloop()
