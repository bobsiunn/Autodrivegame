import cv2
import numpy as np

class Utility():
	def __init__(self):
		self.line_thickness = 2

	def drawLines(self, frame, lines, color):
		try:
			for line in lines:
				cv2.line(frame, (line[0], line[1]), (line[2], line[3]), color, self.line_thickness)
		except IndexError:
			None
	
		return frame

	def drawROILines(self, frame, roi_points):
		cv2.polylines(frame, np.int32([roi_points]), True, (147,20,255), 1)

		return frame

	def showImage(self, frame, title):
		cv2.imshow(title, frame)

	def printDetectedObjects(self, detected_object_list):
		for _object in detected_object_list:
			print(_object.type, "-", _object.pos)