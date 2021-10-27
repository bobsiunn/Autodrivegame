import cv2
import numpy as np
import sys

class LaneDetector():
    def __init__(self, plot_canny = False, plot_binary = False, plot_high_level_binary = False, plot_canny_and_binary = False):
        self.origFrame = None
        self.origFrame_filtered = None
        self.origFrame_blured = None
        self.origFrame_size = None

        self.interested_part_orig = None
        self.interested_binary_image = None
        self.interested_part_blured = None

        self.roi_points = np.array([[(160,250), (0,350), (480, 350), (320, 250)]])

        self.changing_rate = 7
        self.loop_rate = 15
        self.kernal = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]])

        self.extracted_lines = []

        self.plot_canny = plot_canny
        self.plot_binary = plot_binary
        self.plot_high_level_binary = plot_high_level_binary
        self.plot_canny_and_binary = plot_canny_and_binary

    def detectLine(self, frame):
        # Update origFrame
        self.origFrame = frame

        # Process Image
        self.preprocessImage(120)

        # crop image of interested point (Region Of Interested: ROI)
        self.cropImage()

        # get lines
        self.getLines()

        return self.extracted_lines

    def cropImage(self):
        # interested part of orig frame
        target = cv2.cvtColor(self.origFrame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(target)
        cv2.fillPoly(mask, self.roi_points, 255)
        self.interested_part_orig = cv2.bitwise_and(target, mask)
        
        #interestd part of blured orig frame
        target = cv2.cvtColor(self.origFrame_blured, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(target)
        cv2.fillPoly(mask, self.roi_points, 255)
        self.interested_part_blured = cv2.bitwise_and(target, mask)

    def preprocessImage(self, value):
        self.origFrame = cv2.resize(self.origFrame, (480, 480)) 
        self.origFrame_blured = cv2.GaussianBlur(self.origFrame, (5,5), 3)
        self.origFrame_filtered= cv2.filter2D(self.origFrame, -1, self.kernal) #make image more sharper
        self.origFrame_size = self.origFrame.shape[::-1][1:]

    # 기울기 절댓값이 0.3 미만인 직선들 제외 (가로선 제외) (노이즈 제거)
    def lineTreaming(self, lines):
        result = []
        out_of_result = []

        for line in lines:
            if line[2] != line[0]: 
                line_slop = (line[3] - line[1]) / (line[2] - line[0])
            else: 
                line_slop = sys.float_info.max

            if abs(line_slop) > 0.3:
                result.append(line)
        return result

    def getLineCount(self, lines):
        try:
            line_numbers = len(lines)
        except TypeError:
            line_numbers = 0

        return line_numbers

    def getLines(self):
        threshold_level = 130

        # get binary image of blured image of interested part
        (thresh, self.interested_binary_image) = cv2.threshold(self.interested_part_blured, threshold_level, 255, cv2.THRESH_BINARY)
        self.interested_binary_image = cv2.GaussianBlur(self.interested_binary_image, (5,5), 3)

        if self.plot_binary: 
            cv2.imshow("binary Image", self.interested_binary_image)
        
        canny = cv2.Canny(self.origFrame, 45, 200)
        canny_binaryImage = cv2.bitwise_and(canny, self.interested_binary_image)
        hough_lines = cv2.HoughLinesP(canny_binaryImage, 1, np.pi / 180, 75, minLineLength=20, maxLineGap=400)

        line_numbers = self.getLineCount(hough_lines)

        tryCount = 0

        # tryCount가 self.loop_count보다 작으면 while문 안으로 진입하고 line_number가 0보다 크고 8보다 작으면 or try count가 self.loop_rate보다 커지면 탈출
        while tryCount <= self.loop_rate:
            if 2 < line_numbers and line_numbers < 8:
                break

            # 흰색의 비율이 어느정도 이상이면 흰색 비율을 낮추도록 조정 또는 라인 갯수가 몇개 이상이면 비율을 낮추도록 조정
            white_pixels = cv2.countNonZero(self.interested_binary_image)
            white_ratio = (white_pixels/self.interested_binary_image.size) * 100

            if white_ratio > 10 or line_numbers > 8:
                threshold_level += self.changing_rate
                (thresh,self.interested_binary_image) = cv2.threshold(self.interested_part_blured, threshold_level, 255, cv2.THRESH_BINARY)

            elif white_ratio < 1.5 or line_numbers < 2:
                threshold_level -= self.changing_rate
                (thresh,self.interested_binary_image) = cv2.threshold(self.interested_part_blured, threshold_level, 255, cv2.THRESH_BINARY)

            # canny와 higher level of binary image 간의 bitwise and 연산 실행
            self.interested_binary_image = cv2.GaussianBlur(self.interested_binary_image, (5,5), 3)
            canny_binaryImage = cv2.bitwise_and(canny, self.interested_binary_image)
            hough_lines = cv2.HoughLinesP(canny_binaryImage, 1, np.pi / 180, 75,minLineLength=20, maxLineGap=400) 

            tryCount += 1

            line_numbers = self.getLineCount(hough_lines)
              
        
        # 위 과정을 거치고 self.lines에 line들 저장
        lines = []
        try:
            lines = [line[0] for line in hough_lines]
        except TypeError:
            None
        
        self.extracted_lines = self.lineTreaming(lines)

        if self.plot_canny: 
            cv2.imshow("sobel_canny", canny)
        if self.plot_high_level_binary: 
            cv2.imshow("higher binary", self.interested_binary_image)  
        if self.plot_canny_and_binary: 
            cv2.imshow("canny & binary image", canny_binaryImage)

