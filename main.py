import cv2
import numpy as np

from PIL import ImageGrab

from Autodrivegame import drive_utils, objectDetection, laneDetection, car
from Autodrivegame.config import FONT, BLUE, GREEN, RED, YOUTUBE_GRAB_AREA, GRAB_AREA, ROI_POINTS
from Autodrivegame.objectTracker import Tracker

myCar = car.Car()

def start(drive_utils, lane_detector, object_detector):
    while(True):
        # Grab Image of screen
        screen = np.array(ImageGrab.grab(bbox = GRAB_AREA))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        FRAME = cv2.resize(frame, (480, 480))

        line_datas = lane_detector.detectLine(FRAME)

        detected_object_list = object_detector.detectObject(FRAME)

        RESULT_FRAME = drive_utils.displayInfo(FRAME, detected_object_list)
        RESULT_FRAME = drive_utils.drawLines(RESULT_FRAME, line_datas, GREEN)
        RESULT_FRAME = drive_utils.drawROILines(RESULT_FRAME, lane_detector.roi_points)

        # drive_utils.printDetectedObjects(detected_object_list)
        cv2.imshow("result", RESULT_FRAME)
        
        #object_lists = []
        #myCar.drive(lines, object_lists)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    

if __name__ == "__main__":
    # Settings 
    drive_utils = drive_utils.Utility()
    object_tracker = Tracker()
    lane_detector =  laneDetection.LaneDetector(
        ROI_POINTS,
        plot_canny = False, 
        plot_binary = False, 
        plot_high_level_binary = False, 
        plot_canny_and_binary = False
    )
    object_detector = objectDetection.objectDetector(
        trained_model="yolact/weights/yolact_base_54_800000.pth", 
        top_k=25,
        score_threshold=0.4,
        tracker = object_tracker
    )

    
    start(drive_utils, lane_detector, object_detector)