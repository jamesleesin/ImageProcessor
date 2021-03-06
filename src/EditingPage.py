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
import threading

# For image from memory
from io import StringIO
from kivy.core.image.img_pygame import ImageLoaderPygame

import cv2
from matplotlib import pyplot as plt

class EditingScreen(Screen):
	Window.clearcolor = (0.5, 0.5, 0.5, 1) # Background color
	tool_label_text = ObjectProperty(None)
	image_filepath = StringProperty(None)
	status = ObjectProperty()
	edited_image = ObjectProperty()

	loadfile = ObjectProperty()
	saveFile = ObjectProperty()

	# Initialize the analysis screen
	def __init__(self, **kwargs):
		super(EditingScreen, self).__init__(**kwargs)

	########## Loading and saving files ###################
	def saveFile(self, path, filename):
		currentImage = self.edited_image.processed_image
		if (currentImage != None and currentImage.size > 0):
			cv2.imwrite(os.path.join(path,filename), cv2.flip(currentImage, 0))
			self.status.text = "Saved image to " + os.path.join(path,filename)
		else:
			self.status.text = "Failed to save image." 
		self.dismiss_popup()

	# Just copy a version of the file into a folder
	def openFile(self, path, filename):
		if (self.edited_image != None):
			self.image_filepath = os.path.join(path,filename[0])
			self.edited_image.setup()
			self.status.text = "Opened " + self.image_filepath
		else:
			self.status.text = "Failed to open image."
		self.dismiss_popup()

	def dismiss_popup(self):
		self._popup.dismiss()

	def show_load(self):
		content = Factory.LoadDialog(load=self.openFile, cancel=self.dismiss_popup)
		self._popup = Popup(title="Load file", content=content, size_hint=(0.8,0.8))
		self._popup.open()

	def show_save(self):
		content = Factory.SaveDialog(save=self.saveFile, cancel=self.dismiss_popup)
		self._popup = Popup(title="Save file", content=content, size_hint=(0.8,0.8))
		self._popup.open()

	class LoadDialog(FloatLayout):
		load = ObjectProperty(None)
		cancel = ObjectProperty(None)

	class SaveDialog(FloatLayout):
		save = ObjectProperty(None)
		text_input = ObjectProperty(None)
		cancel = ObjectProperty(None)

	############## Undo and Redo ######################
	def undoOperation(self):
		if (self.edited_image.image_stack_index > 0):
			# Can keep undoing
			self.edited_image.image_stack_index -= 1
			self.edited_image.updateImage(self.edited_image.processed_image_stack[self.edited_image.image_stack_index], False, "")
		else:
			print("Nothing to undo")

	def redoOperation(self):
		if (self.edited_image.image_stack_index < len(self.edited_image.processed_image_stack) - 1):
			# Can keep redoing
			self.edited_image.image_stack_index += 1
			self.edited_image.updateImage(self.edited_image.processed_image_stack[self.edited_image.image_stack_index], False, "")
		else:
			# Already at most recent element
			print("Nothing to redo")

	############### Operation History #############
	class ToolHistory(Popup):
		history_container = ObjectProperty()
		def populate(self, historyArray):
			# Clear history first
			for i in range(len(self.history_container.children)):
				self.history_container.remove_widget(self.history_container.children[0])

			# Insert all of the history into the popup
			for elem in historyArray:
				newLabel = Label()
				newLabel.text = elem
				newLabel.font_size = 14
				newLabel.size_hint = (1, None)
				newLabel.height = 20
				self.history_container.add_widget(newLabel)

	############## Image info popup class #################
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
		status_text = ObjectProperty(None)

		# When the user performs an operation on the image, save it into this array. 
		processed_image_stack = []
		processed_image_stack_commands = []
		image_stack_index = -1

		def __init(self, **kwargs):
			super(EditedImage, self).__init__(**kwargs)
		def setup(self):
			if (self.parent.parent.parent.parent.image_filepath != ""):
				# Reset the image stack and index
				self.processed_image_stack = []
				self.processed_image_stack_commands = []
				self.image_stack_index = -1

				# Read in the image. The image is read in flipped, so flip it back
				self.orig_image = cv2.flip(cv2.imread(self.parent.parent.parent.parent.image_filepath), 0)
				self.updateImage(self.orig_image, True, "Loaded Image.")
				self.status_text = self.parent.parent.parent.parent.status

		# Show the operation history
		def showHistory(self):
			self.toolHistory = Factory.ToolHistory()
			self.toolHistory.populate(self.processed_image_stack_commands)
			self.toolHistory.open()

		# Convert image to a string then to a texture and display on image canvas
		@mainthread
		def updateImage(self, im, addToStack, cmd):
			# Update the displayed image
			self.processed_image = im
			data = im.tostring()
			image_texture = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
			image_texture.blit_buffer(data, colorfmt='bgr', bufferfmt='ubyte')
			with self.canvas:
				self.texture = image_texture

			if (addToStack):
				# Look at the image stack index. if it is at a position other than the end, then drop all images
				# past the current index and then append the new image. Else, just append the image.
				if (self.image_stack_index + 1 != len(self.processed_image_stack)):
					while (self.image_stack_index + 1 < len(self.processed_image_stack)):
						self.processed_image_stack.pop()
						self.processed_image_stack_commands.pop()
				# Then append the image onto the end, and update image_stack_index
				self.processed_image_stack.append(self.processed_image)
				self.processed_image_stack_commands.append(cmd)
				self.image_stack_index = len(self.processed_image_stack) - 1

		# Update status text with a thread
		def updateStatusText(self, text):
			self.status_text.text = text

		# Utility function, set the status text to Ready
		def setStatusReady(self):
			updateText = "Ready."
			updateStatusThread = threading.Thread(target=self.updateStatusText, args=(updateText,))
			updateStatusThread.start()

		############ Functions that run the image processing algorithms on threads ###########
		def computeImageInfo(self):
			updateText = "Displaying Basic Image Information"
			updateStatusThread = threading.Thread(target=self.updateStatusText, args=(updateText,))
			updateStatusThread.start()
			computeImageInfoThread = threading.Thread(target=self.runComputeImageInfo)
			computeImageInfoThread.start()

		def changeCLAHE(self, cl, adj, gs):
			if (not (cl == "" or gs == "")):
				if (isNumber(cl) and isNumber(adj) and isNumber(gs)):
					updateText = "Performing CLAHE: clipSize=" + str(cl) + ", adj=" + str(int(adj * 100)/100.0) + ", gridSize=(" + str(gs) + "," + str(gs) + ")"
					updateStatusThread = threading.Thread(target=self.updateStatusText, args=(updateText,))
					updateStatusThread.start()
					claheThread = threading.Thread(target=self.runCLAHE, args=(cl, adj, gs,))
					claheThread.start()
				else:
					self.updateStatusText("Could not perform CLAHE. Grid size must be between 1 and 100.")

		def blur(self, ks, alg):
			if (ks != ""):
				if (isNumber(ks) and int(ks)%2==1 and int(ks)>0):
					updateText = "Performing Blur: kernelSize=(" + ks + "," + ks + ") with algorithm " + alg
					updateStatusThread = threading.Thread(target=self.updateStatusText, args=(updateText,))
					updateStatusThread.start()
					blurThread = threading.Thread(target=self.runBlur, args=(ks, alg,))
					blurThread.start()
				else:
					self.updateStatusText("Could not perform Blur. Blur kernel size must be a positive odd integer value.")

		def colorQuantize(self, ks, nc):
			if (not (ks == "" or nc == "")):
				if (isNumber(ks) and int(ks)%2==1 and int(ks)>0 and isNumber(nc) and int(nc)>0):
					updateText = "Performing Color Quantization: kernelSize=(" + ks + "," + ks + "), clusters=" + nc
					updateStatusThread = threading.Thread(target=self.updateStatusText, args=(updateText,))
					updateStatusThread.start()
					quantThread = threading.Thread(target=self.runColorQuantize, args=(ks,nc,))
					quantThread.start()
				else:
					self.updateStatusText("Could not perform Color Quantization. Blur kernel size must be a positive odd integer value, and # clusters must be > 0.")

		def detailInpaint(self, ks, dil):
			if (isNumber(ks) and int(ks)%2==1 and int(ks)>0 and isNumber(dil) and int(dil)>0):
				updateText = "Performing detail inpainting: kernelSize=(" + ks + ", " + ks + "), dilationSize=" + dil
				updateStatusThread = threading.Thread(target=self.updateStatusText, args=(updateText,))
				updateStatusThread.start()
				detailInpaintThread = threading.Thread(target=self.runDetailInpaint, args=(ks,dil,))
				detailInpaintThread.start()
			else:
				self.updateStatusText("Could not inpaint on details. Kernel Size must be a positive odd integer value, and dilation must be positive.")

		############ Image Processing Algorithms ###################	
		# Calculate some basic image information such as min/max/mean etc.
		def computeImageInfo(self):
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
			self.setStatusReady()

		# Contrast Limited Adaptive Histogram Equilization. Takes in a clip limit, adjustment value, and a gridsize.
		def runCLAHE(self, cl, adj, gs):
			# Process the inputs
			clipLimit = float(cl) if float(cl) > 0.01 else 0.01
			gridSize = int(gs) if int(gs) > 0 and int(gs) < 100 else 1
			adjustment = float(adj)
			image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2LAB)
			l, a, b = cv2.split(image)
			# Apply CLAHE to lightness channel.
			# I think with a tile size of (1,1) it is basically contrast adjustment
			clahe = cv2.createCLAHE(clipLimit * adjustment, tileGridSize=(gridSize,gridSize))
			cl = clahe.apply(l)
			image = cv2.merge((cl, a, b))
			image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
			self.updateImage(image, True, "CLAHE: cl=" + str(clipLimit) + "adj=" + str(int(adj * 100)/100.0) + ", gridSize=(" + str(gs) + "," + str(gs) + ")")
			self.setStatusReady()

		# Blur. Parameters: kernel size and algorithm
		def runBlur(self, ks, alg):
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
			self.updateImage(image, True, "Blur: ks=(" + ks + "," + ks + "), alg=" + alg)
			self.setStatusReady()

		# Color quantization. Optional blur before running the quantization, with numclusters=nc
		def runColorQuantize(self, ks, nc):
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
			self.updateImage(quant, True, "Color Quantization: ks=(" + ks + "," + ks + "), nc=" + nc)
			self.setStatusReady()

		# Inpainting of details (ie. hair removal). Takes in a blur kernel size and a dilation size.
		def runDetailInpaint(self, ks, dil):
			# Blur a copy of the image, then subtract a copy of it from the image
			image = self.orig_image
			blurred = cv2.medianBlur(image, int(ks))
			diff = cv2.subtract(blurred, image)
			# Convert to gray
			grayDiff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
			ret3, thr = cv2.threshold(grayDiff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			dilationConv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*int(dil)+1, 2*int(dil)+1), (int(dil), int(dil)))
			mask = cv2.dilate(thr, dilationConv, 1)
			res = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
			self.updateImage(res, True, "Detail Inpainting: ks=(" + ks + ", " + ks + "), ds=" + dil)
			self.setStatusReady()

# Check to see if a text input is a number
def isNumber(s):
	try:
		float(s)
		return True
	except ValueError:
		return False