from __future__ import print_function

#!/usr/bin/kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.image import AsyncImage
from kivy.factory import Factory
from kivy.uix.popup import Popup
from kivy.clock import mainthread
from kivy.graphics import Rectangle, Color
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.animation import Animation

# images
import numpy as np
from PIL import Image

import os

# Other 
import csv
import time
import shutil
import math

# For image from memory
from io import StringIO
from kivy.core.image.img_pygame import ImageLoaderPygame

import cv2
from matplotlib import pyplot as plt

class EditingScreen(Screen):
	Window.clearcolor = (0.5, 0.5, 0.5, 1) # Background color
	tool_label_text = ObjectProperty(None)
	image_filepath = StringProperty(None)

	# Initialize the analysis screen
	def __init__(self, **kwargs):
		super(EditingScreen, self).__init__(**kwargs)

	class ImageInfo(Popup):
		imageShape = ObjectProperty()
		imageType = ObjectProperty()
		bMinMax = ObjectProperty()
		gMinMax = ObjectProperty()
		rMinMax = ObjectProperty()
		bMeanStdDev = ObjectProperty()
		gMeanStdDev = ObjectProperty()
		rMeanStdDev = ObjectProperty()
		histogramImage = ObjectProperty()

		def __init(self, **kwargs):
			super(ImageInfo, self).__init__(**kwargs)
		def populate(self, info):
			self.imageShape.text = info[0]
			self.imageType.text = info[1]
			self.bMinMax.text = info[2]
			self.gMinMax.text = info[3]
			self.rMinMax.text = info[4]
			self.bMeanStdDev.text = info[5]
			self.gMeanStdDev.text = info[6]
			self.rMeanStdDev.text = info[7]
			# Need to reload the histogram image 
			self.histogramImage.reload()


	class EditedImage(AsyncImage):
		memory_data = ObjectProperty(None)
		orig_image = None
		processed_image = None
		def __init(self, **kwargs):
			super(EditedImage, self).__init__(**kwargs)
		def setup(self):
			# Read in the image. The image is read in flipped, so flip it back
			self.orig_image = cv2.flip(cv2.imread(self.parent.parent.parent.parent.image_filepath), 0)
			self.updateImage(self.orig_image)
		# Convert image to a string then to a texture and display on image canvas
		def updateImage(self, im):
			self.processed_image = im
			data = im.tostring()
			image_texture = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
			image_texture.blit_buffer(data, colorfmt='bgr', bufferfmt='ubyte')
			with self.canvas:
				self.texture = image_texture

		############ Image Info ######################
		# Calculate some basic image information such as min/max/mean etc.
		def computeImageInfo(self):
			print("Displaying Basic Image Information")
			image = self.processed_image
			if (len(image.shape)<3):
				# Grayscale
				min_val = np.amin(image[:,:,0])
				max_val = np.amax(image[:,:,0])
				print("Image Min Value: " + str(min_val))
				print("Image Max Value: " + str(max_val))
			elif (len(image.shape)==3):
				# BGR
				# Split image into channels 
				b,g,r = cv2.split(image)
				bMin,bMax,minLoc,maxLoc = cv2.minMaxLoc(b)
				gMin,gMax,minLoc,maxLoc = cv2.minMaxLoc(g)
				rMin,rMax,minLoc,maxLoc = cv2.minMaxLoc(r)
				bMean,bStdDev = cv2.meanStdDev(b)
				gMean,gStdDev = cv2.meanStdDev(g)
				rMean,rStdDev = cv2.meanStdDev(r)
			else:
				pass

			info = []
			info.append(str(image.shape[1]) + "x" + str(image.shape[0]) + "x" + str(image.shape[2]))
			info.append(str(image.dtype))
			info.append(str(bMin) + " - " + str(bMax))
			info.append(str(gMin) + " - " + str(gMax))
			info.append(str(rMin) + " - " + str(rMax))
			info.append(str(int(bMean[0][0]*100)/100.0) + ", " + str(int(bStdDev[0][0]*100)/100.0))
			info.append(str(int(gMean[0][0]*100)/100.0) + ", " + str(int(gStdDev[0][0]*100)/100.0))
			info.append(str(int(rMean[0][0]*100)/100.0) + ", " + str(int(rStdDev[0][0]*100)/100.0))

			# Calculate Histograms for each channel
			color = ('b', 'g', 'r')
			fig=plt.figure()
			fig.set_size_inches((1,1))
			ax = plt.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)
			for i,col in enumerate(color):
				histr = cv2.calcHist([image], [i], None, [256], [0,256])
				plt.plot(histr, color = col, linewidth=0.75)
				plt.xlim([0,256])
			plt.savefig("hist.png", dpi=600)

			self.imageInfo = Factory.ImageInfo()
			self.imageInfo.populate(info)
			self.imageInfo.open()



		############ Image Processing Algorithms ###################	
		# Contrast Limited Adaptive Histogram Equilization. Takes in a clip limit, adjustment value, and a gridsize.
		def changeCLAHE(self, cl, adj, gs):
			if (not (cl == "" or gs == "")):
				if (isNumber(cl) and isNumber(adj) and isNumber(gs)):
					print("Performing CLAHE: clipSize=" + str(cl) + ", adj=" + str(int(adj * 100)/100.0) + ", gridSize=(" + str(gs) + "," + str(gs) + ")")
					# Process the inputs
					clipLimit = float(cl) if float(cl) > 0.01 else 0.01
					gridSize = int(gs) if int(gs) > 0 and int(gs) < 20 else 1
					adjustment = float(adj)
					image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2LAB)
					l, a, b = cv2.split(image)
					# Apply CLAHE to lightness channel.
					# I think with a tile size of (1,1) it is basically contrast adjustment
					clahe = cv2.createCLAHE(clipLimit * adjustment, tileGridSize=(gridSize,gridSize))
					cl = clahe.apply(l)
					image = cv2.merge((cl, a, b))
					image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
					self.updateImage(image)
				else:
					print("Could not perform CLAHE. Values must be valid numbers.")

		# Blur. Parameters: kernel size and algorithm
		def blur(self, ks, alg):
			if (ks != ""):
				if (isNumber(ks) and int(ks)%2==1 and int(ks)>0):
					print("Performing Blur: kernelSize=(" + ks + "," + ks + ") with algorithm " + alg)
					kernelSize = int(ks)
					image = None
					if (alg == "Box"):
						image = cv2.blur(self.orig_image,(kernelSize, kernelSize))
					elif (alg == "Gaussian"):
						image = cv2.GaussianBlur(self.orig_image,(kernelSize, kernelSize), 0)
					elif (alg == "Median"):
						image = cv2.medianBlur(self.orig_image, kernelSize)
					elif (alg == "Bilateral"):
						image = cv2.bilateralFilter(self.orig_image, kernelSize, -1, -1)
					self.updateImage(image)
				else:
					print("Could not perform Blur. Blur kernel size must be a positive odd integer value.")

		# Color quantization. Optional blur before running the quantization, with numclusters=nc
		def colorQuantize(self, ks, nc):
			if (not (ks == "" or nc == "")):
				if (isNumber(ks) and int(ks)%2==1 and isNumber(nc) and int(nc)>0):
					print("Performing Color Quantization: kernelSize=(" + ks + "," + ks + "), clusters=" + nc)
					image = self.orig_image
					# Optional blur before quantizing
					if (int(ks)>0):
						image = cv2.medianBlur(image, int(ks))

					image = np.float32(image)
					(h, w) = image.shape[:2]
					image = image.reshape((image.shape[0]*image.shape[1], 3))
					# Apply kmeans with the specified number of clusters
					criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
					ret, label, center = cv2.kmeans(image,int(nc),None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
					center = np.uint8(center)
					quant = center[label.flatten()]
					quant = quant.reshape((h,w,3))
					self.updateImage(quant)
				else:
					print("Could not quantize. Number of clusters must be > 0")

		# Inpainting of details (ie. hair removal). Takes in a blur kernel size and a threshold.
		def detailInpaint(self, ks, thr):
			pass

# Check to see if a text input is a number
def isNumber(s):
	try:
		float(s)
		return True
	except ValueError:
		return False