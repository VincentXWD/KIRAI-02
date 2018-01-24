import sys
import os
from PyQt4 import QtGui,QtCore
import cv2
from PIL import Image
import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class Window(QtGui.QDialog):
	"""docstring for Window"""
	def __init__(self,parent=None):
		super(Window, self).__init__(parent)
		self.setGeometry(50,50,500,500)
		self.setWindowTitle("Brain Image Segmentation")
		#self.setWindowIcon(QtGui.QIcon('logo.png'))

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.button = QtGui.QPushButton('Original Image')
		self.button1 = QtGui.QPushButton('Blurred Image')
		self.button2 = QtGui.QPushButton('White Matter')
		self.button3 = QtGui.QPushButton('Grey Matter')
		self.button4 = QtGui.QPushButton('White and Grey Matter')
		self.button.clicked.connect(self.originalImage)
		self.button1.clicked.connect(self.blurredImage)
		self.button2.clicked.connect(self.overlayBlurAndWhitMatter)
		self.button3.clicked.connect(self.overlayBlurAndGreyMatter)
		self.button4.clicked.connect(self.imageWhiteAndGrey)

		layout = QtGui.QVBoxLayout()
		layout.addWidget(self.toolbar)
		layout.addWidget(self.canvas)
		layout.addWidget(self.button)
		layout.addWidget(self.button1)
		layout.addWidget(self.button2)
		layout.addWidget(self.button3)
		layout.addWidget(self.button4)
		self.setLayout(layout)

		#self.setLayout(self.main_layout)


		pathDicom = "MyHead/"
		idxSlice = 50
		self.labelWhiteMatter = 1
		self.labelGrayMatter = 2
		reader = SimpleITK.ImageSeriesReader()						#read all the files
		filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)	#save the file names into an array
		reader.SetFileNames(filenamesDICOM)							#save the list of dicom images in an original image array
		self.imgOriginal = reader.Execute()
		self.imgOriginal = self.imgOriginal[:,:,idxSlice]
		self.imgSmooth = SimpleITK.CurvatureFlow(image1=self.imgOriginal,timeStep=0.125,numberOfIterations=5)	#function for converting smooth image array

	
	#function for converting all the discom files into an single image
	def sitk_show(self,img, title=None, margin=0.05, dpi=40):
		nda = SimpleITK.GetArrayFromImage(img)						#get an array from the original image(array)
		spacing = img.GetSpacing()
		figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
		extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
		fig = plt.figure(figsize=figsize, dpi=dpi)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		plt.set_cmap("gray")
		ax.imshow(nda,extent=extent,interpolation=None)				#showing the image
		self.canvas.draw()

	#function for showing the original image on clicked original imafge
	def originalImage(self):
		self.sitk_show(self.imgOriginal)

	#function for converting blurry image
	def blurredImage(self):
		self.sitk_show(self.imgSmooth)

	#function for white matter
	def overlayBlurAndWhitMatter(self):		
		lstSeeds = [(150  ,75)]
		imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=self.imgSmooth,seedList=lstSeeds,lower=130,upper=190,replaceValue=self.labelWhiteMatter)
		imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(self.imgSmooth), imgWhiteMatter.GetPixelID())
		imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=self.labelWhiteMatter)
		self.sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatterNoHoles))

	def overlayBlurAndGreyMatter(self):
		lstSeeds = [(119, 83), (198, 80), (185, 102), (164, 43)]
		imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=self.imgSmooth,seedList=[(150,75)],lower=130,upper=190,replaceValue=self.labelWhiteMatter)
		imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(self.imgSmooth), imgWhiteMatter.GetPixelID())
		imgGrayMatter = SimpleITK.ConnectedThreshold(image1=self.imgSmooth,seedList=lstSeeds,lower=150,upper=270,replaceValue=self.labelGrayMatter)
		imgGrayMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGrayMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=self.labelGrayMatter)
		self.sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgGrayMatterNoHoles))

	def imageWhiteAndGrey(self):
		lstSeeds = [(119, 83), (198, 80), (185, 102), (164, 43)]
		imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=self.imgSmooth,seedList=[(150,75)],lower=130,upper=190,replaceValue=self.labelWhiteMatter)
		imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(self.imgSmooth), imgWhiteMatter.GetPixelID())
		imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=self.labelWhiteMatter)
		imgGrayMatter = SimpleITK.ConnectedThreshold(image1=self.imgSmooth,seedList=lstSeeds,lower=150,upper=270,replaceValue=self.labelGrayMatter)
		imgGrayMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGrayMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=self.labelGrayMatter)
		imgLabels = imgWhiteMatterNoHoles | imgGrayMatterNoHoles
		self.sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgLabels))




def run():
	app=QtGui.QApplication(sys.argv)
	GUI=Window()
	GUI.show()
	sys.exit(app.exec_())


run()
		