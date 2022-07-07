import cv2
from transform import getPerspective
import numpy as np
from imutils import resize

def resize(image,height):
	(h,w) = image.shape[:-1]
	ratio = height / float(h)
	aspect_ratio = h / float(height)
	resized = cv2.resize(image,(int(w*ratio), height),interpolation=cv2.INTER_AREA)
	return aspect_ratio,resized

def findPage(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	canny = cv2.Canny(gray,100,200)
	cnts = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
	cnts = max(cnts,key = cv2.contourArea)
	cnts = cv2.approxPolyDP(cnts, 0.02*cv2.arcLength(cnts,True), True)

	if len(cnts) == 4:
		page = cnts
	else:
		page = None
	return page

def sharpen(image):
	kernel = np.array([[0, -1, 0],
			[-1, 5,-1],
			[0, -1, 0]])
	sharped = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
	return sharped


if __name__ == "__main__":
	image = cv2.imread("Images\\img2.jpeg")
	aspect_ratio,reshaped = resize(image,500)
	page = findPage(reshaped)
	if page is not None:
		result = getPerspective(image,np.squeeze(page)*aspect_ratio)
		sharped = sharpen(result)
		cv2.imshow("Result",result)
		cv2.imshow("Sharped",sharped)
		cv2.waitKey(0)
	else:
		print("Sorry! an error has occured")