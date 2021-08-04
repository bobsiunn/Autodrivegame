import os
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image
import threading

from Autodrivegame import objectDetection, laneDetection, myUtils, car

FRAME = None
OBJECT_FRAME = None
LINES = []
FINISH_SIGN = False

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAB_AREA = (0, 0, 1100, 800)

drive_utils = myUtils.Utility()

def getLineAndShow():
    global FRAME
    global FINISH_SIGN
    global LINES

    while True:
        if FINISH_SIGN: return
        LINES = lane_detector.detectLine(FRAME)
        time.sleep(0.1)

if __name__ == "__main__":

    lane_detector =  laneDetection.LaneDetector(
        plot_canny = False, 
        plot_binary = False, 
        plot_high_level_binary = False, 
        plot_canny_and_binary = False
    )
    myCar = car.Car()
    object_detector = objectDetection.objectDetector(
        trained_model="myYolact/weights/yolact_base_54_800000.pth", 
        top_k=15,
        score_threshold=0.4
    )

    # lane detection은 thread로 처리
    t1 = threading.Thread(target=getLineAndShow, args=())

    First = True
    while(True):
        # Grab Image of screen
        screen = np.array(ImageGrab.grab(bbox = GRAB_AREA))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        FRAME = cv2.resize(frame, (480, 480))

        if First:
            t1.start()
            First = False

        OBJECT_FRAME = object_detector.detectObject(FRAME)

        RESULT_FRAME = drive_utils.drawLines(OBJECT_FRAME, LINES, GREEN)

        drive_utils.showImage(RESULT_FRAME, "result")

        #object_lists = []
        #myCar.drive(lines, object_lists)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            FINISH_SIGN = True

            t1.join()

            cv2.destroyAllWindows()
            break
    