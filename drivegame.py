import numpy as np
#이미지 캡쳐 위한 imageGrab
from PIL import ImageGrab
#이미지 처리 위한 opencv
import cv2
#실행시간 실시간 측정(속도 측정 등에 쓰일 것으로 예상)
import time

#실행시간 저장 변수 last_time을 실행 시작 시간으로 초기화
last_time = time.time()

while(True):
    #화면 기준 (0,0) 픽셀부터 (800,640) 픽셀까지 캡쳐하고 배열로 전환
    screen = np.array(ImageGrab.grab(bbox = (0, 0, 800, 640)))
    #실행시간 출력
    print('Loop took {} seconds'.format(time.time()-last_time))
    #실행 시작 시간 업데이트
    last_time = time.time()
    #캡쳐한 화면을 window 창으로 띄우고, 그 색을 RGB로 전환
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    
    #1초 간 키보드 입력을 기다리고, 'q'를 눌렀을 시 모든 창을 닫으며 프로그램 종료
    if(cv2.waitKey(1000) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break