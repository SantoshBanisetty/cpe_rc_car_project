#!/usr/bin/env python
from __future__ import print_function
import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np

from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import matplotlib.pyplot as plt

first = True
sendFit1 = []
SendFit2 = []


class image_converter:

	def __init__(self):
		self.image_pub = rospy.Publisher("image_topic_2",Image, queue_size=10)

		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

	def getImage(self):
		return cv2.imread("test.jpg")
	def toGreyscale(self, clr_image):
		return cv2.cvtColor(clr_image, cv2.COLOR_BGR2GRAY)
	def whiteDetection(self, rgb_image):
		hls_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HLS)
		#cv2.imshow("hls", hls_image)
		white_upr = np.array([255, 255, 255])
		white_lwr = np.array([0, 200, 0])
		white_mask = cv2.inRange(hls_image, white_lwr, white_upr)
		#cv2.imshow("white mask", white_mask)
		white_image = cv2.bitwise_and(rgb_image, rgb_image, mask = white_mask)
		return white_image
	def getBinary(self, ch_img):
		grey = self.toGreyscale(ch_img)
		red_min, red_max = 210, 255
		r, g, b = cv2.split(ch_img)
		sum = r+g+b
		binary = np.zeros_like(grey)
		binary [sum >= 50] = 1
		return binary
	def sobelOperator(self, gray_image):
		return sobel_edge_image
	def combinedBinaryImage(self, image1, image2):
		return binary_image
	def birdsEyeView(self, car_view, pts1, pts2):
		
		M = cv2.getPerspectiveTransform(pts1,pts2)
		bird_view = cv2.warpPerspective(car_view,M,(1279, 951))
		return bird_view

	def getCenterEstimates(self, temp):
		#plt.subplot(233)
		xcount = np.sum(temp[500:], axis=0)
		#plt.plot(xcount)
		#plt.xlabel("x position")
		#plt.ylabel("# white pixels")
		#plt.show()
		c1x = np.argmax(xcount[:640])
		c2x = np.argmax(xcount[640:]) + 640
		return c1x, c2x

	def curvature(self, fit, y):
		A, B = fit[0], fit[1]
		return (1+(2*A*y+B)**2)**1.5/np.absolute(2*A)

	def firstFrame(self, bin_img, cx1, cx2):
		nwindows = 9
		window_height = np.int(bin_img.shape[0]/nwindows)  # 80
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = bin_img.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100 # Set the width of the windows +/- margin
		minpix = 50 # Set minimum number of pixels found to recenter

		# Create empty lists to receive left and right lane pixel indices
		lane1 = []
		lane2 = []
		win_img = np.dstack((bin_img, bin_img, bin_img))*255 # prepare image to draw window

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			y_min = bin_img.shape[0] - (window+1)*window_height
			y_max = bin_img.shape[0] - window*window_height
			x1_min, x1_max = cx1 - margin, cx1 + margin
			x2_min, x2_max = cx2 - margin, cx2 + margin
			# Draw the windows 
			cv2.rectangle(win_img,(x1_min,y_min),(x1_max,y_max),(0,0,255), 6) 
			cv2.rectangle(win_img,(x2_min,y_min),(x2_max,y_max),(0,0,255), 6) 
			# Identify the nonzero pixels in x and y within the window
			select1 = ((nonzeroy >= y_min) & (nonzeroy < y_max) & (nonzerox >= x1_min) & (nonzerox < x1_max)).nonzero()[0]
			select2 = ((nonzeroy >= y_min) & (nonzeroy < y_max) & (nonzerox >= x2_min) & (nonzerox < x2_max)).nonzero()[0]
			# Append these indices to the lists
			lane1.append(select1)
			lane2.append(select2)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(select1) > minpix:
				cx1 = np.int(np.mean(nonzerox[select1]))
			if len(select2) > minpix:        
				cx2 = np.int(np.mean(nonzerox[select2]))

		# Concatenate the arrays of indices
		lane1_index = np.concatenate(lane1)
		lane2_index = np.concatenate(lane2)

		# Extract left and right line pixel positions
		x1, y1  = nonzerox[lane1_index], nonzeroy[lane1_index] 
		x2, y2  = nonzerox[lane2_index], nonzeroy[lane2_index] 

		# Fit a second order polynomial to each
		fit1 = np.polyfit(y1, x1, 2)
		fit2 = np.polyfit(y2, x2, 2)

		# Generate x and y values for plotting
		ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0] )
		fitx1 = fit1[0]*ploty**2 + fit1[1]*ploty + fit1[2]  # generate points by fit coeff
		fitx2 = fit2[0]*ploty**2 + fit2[1]*ploty + fit2[2]
		win_img[y1,x1] = [255, 0 , 0]  # factual points
		win_img[y2,x2] = [0,  255, 0]
		# #plt.figure(figsize=(10,5))
		# plt.subplot(235)
		# plt.imshow(win_img)
		# plt.plot(fitx1, ploty, color='white')
		# plt.plot(fitx2, ploty, color='white')
		# plt.title("Window Search")
		# #plt.show()
		# plt.savefig("lanes.png")
		return fit1, fit2

	def pipeline(self, imageP, src, dst, fit1, fit2):
		print ("=============== pipeline")
		print (fit1, fit2)
		#-----------------------------------------------------

		# imageP = getImage()
		dsttP = self.birdsEyeView(imageP, src, dst)
		whiteP = self.whiteDetection(dsttP)
		bin_imgP = self.getBinary(whiteP)

		# plt.figure(2)
		# #plt.subplot(236)
		# plt.imshow(bin_imgP)

		nonzero = bin_imgP.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100 # Set the width of the windows +/- margin
		#minpix = 50 # Set minimum number of pixels found to recenter

		fitx1 = fit1[0]*(nonzeroy**2) + fit1[1]*nonzeroy + fit1[2] # x value on the curve for every nonzero point
		fitx2 = fit2[0]*(nonzeroy**2) + fit2[1]*nonzeroy + fit2[2]
		lane1_index = ((nonzerox > fitx1 - margin) & (nonzerox < fitx1 + margin)) 
		lane2_index = ((nonzerox > fitx2 - margin) & (nonzerox < fitx2 + margin)) 

		# extract left and right line pixel positions
		x1,y1 = nonzerox[lane1_index], nonzeroy[lane1_index]
		x2,y2 = nonzerox[lane2_index], nonzeroy[lane2_index]
		# New fitting parameters
		fit1 = np.polyfit(y1, x1, 2)
		fit2 = np.polyfit(y2, x2, 2)

		# Generate x and y values for plotting
		ploty = np.linspace(0, bin_imgP.shape[0]-1, bin_imgP.shape[0] )
		# Generate x and y values using new fit coeff
		fitx1 = fit1[0]*ploty**2 + fit1[1]*ploty + fit1[2]  # generate points by fit coeff
		fitx2 = fit2[0]*ploty**2 + fit2[1]*ploty + fit2[2]

		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((bin_imgP, bin_imgP, bin_imgP))*255
		out_img[y1,x1] = [0, 0 , 255]  # factual points
		out_img[y2,x2] = [0,  255, 0]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		window_img = np.zeros_like(out_img)
		left_line_window1 = np.array([np.transpose(np.vstack([fitx1-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx1+margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([fitx2-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx2+margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
		# plt.imshow(result)
		# plt.plot(fitx1, ploty, color='blue')
		# plt.plot(fitx2, ploty, color='blue')
		# plt.title("oriented search")
		cv2.namedWindow("OrientedSearch", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("OrientedSearch", 600, 600)
		cv2.imshow("OrientedSearch", result)
		cv2.waitKey(1)

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 2.6924/1280# meters per pixel in y dimension
		xm_per_pix = 0.5334/960 # meters per pixel in x dimension
		# calibrated coefficients 
		print (ym_per_pix, xm_per_pix)
		fit1_cal = np.polyfit(ploty*ym_per_pix, fitx1*xm_per_pix, 2)
		fit2_cal = np.polyfit(ploty*ym_per_pix, fitx2*xm_per_pix, 2)
		# Calculate the new radii of curvature
		curv1 = self.curvature(fit1_cal, 3) # 720 * ym_per_pix
		curv2 = self.curvature(fit2_cal, 3)
		print("Curvatures for left lane: {0:.0f} m, right lane: {1:.0f} m".format(curv1,curv2))

		off_center = -(fitx1[-1]+fitx2[-1]-1280)/2 *xm_per_pix  # assume camera sits in the center of car
		print("The car is right to the center by {0:.2f} m".format(off_center))

		text1 = "Lane Curvatures, left: {0:.0f} m, right: {1:.0f} m".format(curv1,curv2)
		text2 = "Car is right to the center by {0:.2f} m".format(off_center)
		texted_image =cv2.putText(img=np.copy(imageP), text= text1, org=(50,50),
		                          fontFace=3, fontScale=1, color=(0,0,255), thickness=2)
		cv2.putText(img=texted_image, text= text2, org=(50,100),
		                          fontFace=3, fontScale=1, color=(0,0,255), thickness=2)
		# plt.figure(3)
		# plt.imshow(texted_image)
		# Create an image to draw the lines on
		color_warp = np.zeros((960,1280,3)).astype(np.uint8) 

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts1 = np.array([np.transpose(np.vstack([fitx1, ploty]))])
		pts2 = np.array([np.flipud(np.transpose(np.vstack([fitx2, ploty])))])
		pts = np.hstack((pts1, pts2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
		Minv = cv2.getPerspectiveTransform(dst, src)
		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(color_warp, Minv, (texted_image.shape[1], texted_image.shape[0])) 
		# Combine the result with the original image
		result2 = cv2.addWeighted(texted_image, 1, newwarp, 0.3, 0)

		# plt.figure(4)
		# plt.imshow(result)
		# plt.show()

		cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Result", 600, 600)
		cv2.imshow("Result", result2)
		cv2.waitKey(1)


	def callback(self,data):
		global first, sendFit1, SendFit2	
		print ("=============== callback")
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
	
		#src = np.float32([[140, 470],[500,470],[0,479],[639,479]])
		#dst = np.float32([[0,0],[639,0],[0,479],[639, 479]])


		# src = np.float32([[140, 500],[1040,500],[0,790],[1270,790]])
		# dst = np.float32([[0,0],[1279,0],[0,951],[1279, 951]])
		
		src = np.float32([[370, 200],[990,200],[65,780],[1254,780]])
		dst = np.float32([[0,0],[1279,0],[0,951],[1279, 951]])		


		if first == True:
			print ("============ First frame")
			print (cv_image.shape)

			dstt = self.birdsEyeView(cv_image, src, dst)
			white = self.whiteDetection(dstt)
			bin_img = self.getBinary(white)
			cv2.imshow("Image window", white)
			print (dstt.shape)
			cx1, cx2 = self.getCenterEstimates(bin_img)
			print ("Initial center estimate X", cx1, cx2)

			fit1, fit2 = self.firstFrame(bin_img, cx1, cx2)
			print ("At first: ", fit1, fit2)
			sendFit1 = fit1
			SendFit2 = fit2
			cv2.waitKey(1)
			first = False
		else:
			self.pipeline(cv_image, src, dst, sendFit1, SendFit2)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

def main(args):
	ic = image_converter()
	rospy.init_node('image_converter', anonymous=True)
   	
	print ("Running .....")
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
