from rulerdetection import Detector_Yolov4
from pattern import EPattern
import cv2


class WaterLevel:
    def __init__(self):
        self.classMap = {0: 'ruler'}
        self.detector = Detector_Yolov4('models/', self.classMap)
        self.pattern = EPattern()

    def calculateWaterLevel(self, image, rulerFootHeight=0, rulerHeight=100, threeLinesHeight=5):
        # Detect ruler
        res, ges = self.detector.detect(image)

        # Get ruler croped
        ruler = res[0][4]
        # Ruler floating height = |bottom - top|
        rulerHeightPixel = abs(res[0][3] - res[0][1])

        # Detect three line
        result, threeLinesHeightPixel = self.pattern.detect(ruler)

        # Calculate water level
        rulerHeightFloat = (threeLinesHeight *
                            rulerHeightPixel) / threeLinesHeightPixel
        waterLevel = round(rulerHeight - rulerHeightFloat + rulerFootHeight, 3)
        text_cord = (res[0][0]+70, res[0][1]+30)
        cv2.putText(image, str(round(waterLevel, 2)), text_cord, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
        return ges, waterLevel, round(rulerHeightFloat, 2)
