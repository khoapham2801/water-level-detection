from rulerdetection import Detector_Yolov3
from rulerdetection import Detector_Yolov4
from pattern import EPattern
import cv2

rulerFootHeight = 0
rulerHeight = 100
threeLinesHeight = 5
classMap = {0: 'ruler'}
pattern = EPattern()
new_detector = Detector_Yolov4("models/", classMap)
img1 = cv2.imread("F:/vinAI/water-level/data/frames/vid1/frame115200.jpg")
img2 = cv2.imread("F:/vinAI/water-level/data/frames/vid2/frame19.jpg")
img3 = cv2.imread("F:/vinAI/water-level/data/vertical_remove_background.jpg")
img_DongHoi = cv2.imread(
    "F:/vinAI/water-level/data/frames/DongHoi_7/frame1.jpg")
res, ges = new_detector.detect(img1)
crop_img = ges[res[0][1]:res[0][3], res[0][0]:res[0][2]]
cv2.imwrite("F:/vinAI/water-level/data/ruler.jpg", crop_img)
# Get ruler cropped
ruler = res[0][4]
# Ruler floating height = |bottom - top|
rulerHeightPixel = abs(res[0][3] - res[0][1])
# Detect three line
result, threeLinesHeightPixel = pattern.detect(ruler)
# Calculate water level
# print(threeLinesHeightPixel)
rulerHeightFloat = rulerHeightPixel * \
    (threeLinesHeight / threeLinesHeightPixel)
waterLevel = round(rulerHeight - rulerHeightFloat + rulerFootHeight, 3)
print(rulerHeightPixel, threeLinesHeightPixel, rulerHeightFloat, waterLevel)
# print(res[0][1])
cv2.imwrite("F:/vinAI/water-level/data/result.jpg", ges)
cv2.imshow('ges', ges)
cv2.waitKey(0)
cv2.destroyAllWindows()
