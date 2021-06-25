import numpy as np
import argparse
import imutils
import os
import glob
import cv2

TEMPLATE_PATH = "pattern/template"


class EPattern():
    def __init__(self):
        self.templatePath = TEMPLATE_PATH
        # self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        # self.template = cv2.Canny(self.template, 50, 200)
        # (self.tH, self.tW) = self.template.shape[:2]

    def detect(self, img):
        # cv2.imshow('img', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = None
        # loop over each template
        for template_file in os.listdir(self.templatePath):
            # print(template_file)
            template = cv2.imread(TEMPLATE_PATH+"/"+template_file)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template = cv2.Canny(template, 50, 200)
            (tH, tW) = template.shape[:2]
            # for each template, loop over the scales of the image
            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                # resize the image according to the scale, and keep track
                # of the ratio of the resizing
                resized = imutils.resize(
                    gray, width=int(gray.shape[1] * scale))
                r = gray.shape[1] / float(resized.shape[1])

                # if the resized image is smaller than the template, then break
                # from the loop
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

                # detect edges in the resized, grayscale image and apply template
                # matching to find the template in the image
                edged = cv2.Canny(resized, 50, 200)
                result = cv2.matchTemplate(
                    edged, template, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                # if we have found a new maximum correlation value, then update
                # the bookkeeping variable
                # print(maxVal, maxLoc, r)
                if found is None or maxVal > found[0]:
                    best_tH = tH
                    best_tW = tW
                    best_r = r
                    found = (maxVal, maxLoc, best_r, best_tH, best_tW)
            # print(found)

        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        if found is None:
            print("No matching template!")
            return
        (_, maxLoc, r, best_tH, best_tW) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + best_tW)
                            * r), int((maxLoc[1] + best_tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

        return img, abs(endY - startY)
