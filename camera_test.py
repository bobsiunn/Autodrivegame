

from Autodrivegame import objectDetection, laneDetection, myUtils, car

import cv2
import sys




if __name__ == "__main__":
    object_detector = objectDetection.objectDetector(
        trained_model="myYolact/weights/yolact_base_54_800000.pth", 
        top_k=15,
        score_threshold=0.4
    )


    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
    
        ret, frame = capture.read()

        result = object_detector.detectObject(frame)
        cv2.imshow("result", result)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    