import os
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image
import threading

from Autodrivegame import objectDetection, laneDetection, myUtils, car

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAB_AREA = (0, 0, 1100, 800)

drive_utils = myUtils.Utility()

if __name__ == "__main__":

    lane_detector =  laneDetection.LaneDetector(
        plot_canny = False, 
        plot_binary = False, 
        plot_high_level_binary = False, 
        plot_canny_and_binary = False
    )
    object_detector = objectDetection.objectDetector(
        trained_model="yolact/weights/yolact_base_54_800000.pth", 
        top_k=25,
        score_threshold=0.4
    )
    myCar = car.Car()

    while(True):
        # Grab Image of screen
        screen = np.array(ImageGrab.grab(bbox = GRAB_AREA))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        FRAME = cv2.resize(frame, (480, 480))

        LINES = lane_detector.detectLine(FRAME)
        detected_object_list, OBJECT_FRAME = object_detector.detectObject(FRAME)

        RESULT_FRAME = drive_utils.drawLines(OBJECT_FRAME, LINES, GREEN)
        RESULT_FRAME = drive_utils.drawROILines(RESULT_FRAME, lane_detector.roi_points)

        drive_utils.printDetectedObjects(detected_object_list)
        drive_utils.showImage(RESULT_FRAME, "result")

        #object_lists = []
        #myCar.drive(lines, object_lists)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    