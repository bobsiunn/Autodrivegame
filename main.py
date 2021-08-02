import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image

from Autodrivegame import laneDetection, myUtils, car

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GRAB_AREA = (0, 0, 1100, 800)

if __name__ == "__main__":
    lane_detector =  laneDetection.LaneDetector(
        plot_canny = True, 
        plot_binary = True, 
        plot_high_level_binary = True, 
        plot_canny_and_binary = True
    )
    drive_utils = myUtils.Utility()
    myCar = car.Car()

    while(True):
        # Grab Image of screen
        screen = np.array(ImageGrab.grab(bbox = GRAB_AREA))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (480, 480))

        lines = lane_detector.detectLine(frame)

        frame = drive_utils.drawROILines(frame, lane_detector.roi_points)
        frame = drive_utils.drawLines(frame, lines, GREEN)

        drive_utils.showImage(frame, "result")

        object_lists = []
        myCar.drive(lines, object_lists)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    