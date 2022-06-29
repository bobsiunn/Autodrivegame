import cv2
import numpy as np
from Autodrivegame.config import COLORS, FONT, FONT_SCALE, FONT_THICKNESS

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

	def displayInfo(self, frame, detected_object_list, display_bboxes=True, display_text=True, display_pos=True, display_track_num=True, display_scores=True):
		for _object in detected_object_list:
			x1, y1, x2, y2 = _object.pos[:]
			color = COLORS[_object.typeid % len(COLORS)]
			score = _object.score
				
			if display_bboxes:
				cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

			if display_text:
				if display_track_num:
					text_str = '%s[%d]: %.2f' % (_object.type, _object.id, score) if display_scores else (_object.type, _object.id)
				else:
					text_str = '%s: %.2f' % (_object.type, score) if display_scores else _object.type
				if display_pos:
					text_str += (np.array2string(np.array(_object.pos)))
    				

				text_w, text_h = cv2.getTextSize(text_str, FONT, FONT_SCALE, FONT_THICKNESS)[0]
				text_pt = (x1, y1 - 10)

				cv2.rectangle(frame, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
				cv2.putText(frame, text_str, text_pt, FONT, FONT_SCALE, [255, 255, 255], FONT_THICKNESS, cv2.LINE_AA)
		return frame

	def printDetectedObjects(self, object_datas):
		if len(object_datas) == 0: return

		print("========================")
		for _object in object_datas:
			print(f"{_object.type}[{_object.id}] {_object.pos} - {_object.score:.3f}")
		print("========================\n")	