# import packages
from imagesearch.recognizeshape import RecognizeShape
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt

# load image and resize it to a smaller size
image = cv2.imread("geo_shapes.png")
resized = imutils.resize(image, width=400)
ratio = image.shape[0] / float(resized.shape[0])

# convert image to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("rgb2gray", gray)

# histogram of grayscale
histogram = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.plot(histogram)
plt.xlim([0,256])
plt.show()

# gaussian blur of grayscale
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("GaussianBlur",blurred)

# edge detection of blurred image
edges = cv2.Canny(blurred,100,500)
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])
plt.show()

# threshoding of blurred image
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Threshold",thresh)

# find contours in thresholded image and initialize shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = RecognizeShape()

# loop over the contours
for c in cnts:
	# compute center of contour, then detect name of shapes using contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio, then draw the contours and the name shapes on image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
