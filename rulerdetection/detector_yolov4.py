# from .mobilenet import MobileNetMod
from .yolov4 import Yolo4
import cv2
import numpy as np
import base64
from PIL import Image
import io


class Detector_Yolov4():
    def __init__(self, modelPath, classMap):
        self.modelPath = modelPath
        yoloModel = self.modelPath + 'yolov4-custom_final.weights'
        yoloCfg = self.modelPath + 'yolov4-custom.cfg'
        # mobilModel = self.modelPath + 'mobilenet.h5'
        self.yolo4 = Yolo4(yoloModel, yoloCfg)
        # self.mobilenet = MobileNetMod(classMap)
        # self.mobilenet.loadModel(mobilModel)

    def detect(self, frame):
        boxes = self.yolo4.detect(frame)
        res = []
        for box in boxes:
            (_, _, _left, _top, width, height) = box
            left, top, right, bottom = _left, _top, _left + width, _top + height

            if bottom > top and right > left:
                roi = frame[top:bottom, left:right]
                label = 'ruler'  # self.mobilenet.predict(roi)[0]
                res.append((left, top, right, bottom, roi, label, 1))
        return res, self.drawBounding(frame, res)

    def dectectFromBase64(self, base64Img):
        img = self.stringToImg(base64Img)
        return self.detect(img)

    def detectFromByteString(self, byteStringImg):
        nparr = np.fromstring(byteStringImg, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self.detect(img)

    def stringToImg(self, base64Img):
        imgdata = base64.b64decode(str(base64Img))
        image = Image.open(io.BytesIO(imgdata))
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    def drawBounding(self, img, boxes):
        for box in boxes:
            left, top, right, bottom, roi, label, _conf = box
            cv2.rectangle(img, (left, top), (right, bottom), (255, 178, 50), 1)
            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            # cv2.rectangle(img, (left, top - round(1.5*labelSize[1])), (left + round(
            #     1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
            # cv2.putText(img, label, (left, top),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        return img
