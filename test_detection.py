from Autodrivegame import objectDetection, drive_utils, objectTracker
import cv2

if __name__ == "__main__":
    object_detector = objectDetection.objectDetector(
        trained_model="yolact/weights/yolact_base_54_800000.pth", 
        top_k=25,
        score_threshold=0.4,
    )
    myUtils = drive_utils.Utility()
    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = capture.read()

        detected_object_list = object_detector.detectObject(frame)

        result = myUtils.displayInfo(frame, detected_object_list, display_track_num=False) #if display_track_num true, it will show every object 0
        
        # utils.printDetectedObjects(detected_object_list)
        cv2.imshow("result", result)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
    