import numpy as np
import cv2


def ordered(points):
	s = points.sum(axis=1)
	diff = np.diff(points, axis=1)
	top_left = points[np.argmin(s)]
	bottom_right = points[np.argmax(s)]
	top_right = points[np.argmin(diff)]
	bottom_left = points[np.argmax(diff)]
	return np.float32([top_left, top_right, bottom_right, bottom_left])


def getPerspective(image, points):
	input_points = ordered(points)	
	A, B, C, D = input_points
	widthA = np.sqrt(((C[0] - D[0]) ** 2) + ((C[1] - D[1]) ** 2))
	widthB = np.sqrt(((B[0] - A[0]) ** 2) + ((B[1] - A[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((B[0] - C[0]) ** 2) + ((B[1] - C[1]) ** 2))
	heightB = np.sqrt(((A[0] - D[0]) ** 2) + ((A[1] - D[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	
	output_points = np.array(
		[[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
		dtype=np.float32,
	)
	M = cv2.getPerspectiveTransform(input_points, output_points)
	return cv2.warpPerspective(image, M, (maxWidth, maxHeight))
