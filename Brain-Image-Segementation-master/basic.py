#import necessary packages
import numpy as np
import os
import sys
import dicom
from PyQt4 import QtGui,QtCore
import cv2
from PIL import Image
import SimpleITK
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt

#Issues:second Button the one labeled build image from directory is showing some issues.Be suretu turn off the program
#and then restart it again 

class Window(QtGui.QDialog):
	def __init__(self,parent=None):
		super(Window, self).__init__(parent)
		
		self.setGeometry(100,100,950,400)							#window size-first two are co ordinates of where it is starting,second two are size)
		self.setWindowTitle("Brain Image Segmentation")				#window title
		
		#initializing canvas for matplotlib to show
		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		#reading the images and converting them to grayscale
		self.img=cv2.imread('normal.jpg')
		self.grayscale=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)

		#initializing buttons and canvas

		#buttons for main functions
		self.buttonOpenDicomFile = QtGui.QPushButton('Open Single Dicom File')
		self.buttonShowImageNormal = QtGui.QPushButton('Build Image from a directory')

		#buttons for opencv functions
		self.buttonguassianThresh = QtGui.QPushButton('Gausssian Threshhold')
		self.buttonGaussianBlur = QtGui.QPushButton('Gaussian Blur')
		self.buttonBilateralFilter = QtGui.QPushButton('Bilateral Image Filter')
		self.buttonMedianBlur = QtGui.QPushButton('Median Blur')
		self.buttonwatershed = QtGui.QPushButton('Final output after watershed')

		#buttons for simpleITK functions
		self.buttonSITKblur = QtGui.QPushButton('Blur(Median+gaussian)')
		self.buttonSITKgrey = QtGui.QPushButton('Grey Matter')
		self.buttonSITKwhite = QtGui.QPushButton('White Matter')
		self.buttonSITKfinal = QtGui.QPushButton('Final Output')

		#functions when button is clicked
		self.buttonOpenDicomFile.clicked.connect(self.readSingleDicom)
		self.buttonShowImageNormal.clicked.connect(self.showImageNormal)

		self.buttonguassianThresh.clicked.connect(self.gaussianThreshhold)
		self.buttonGaussianBlur.clicked.connect(lambda:self.gaussianFilter(self.img))
		self.buttonBilateralFilter.clicked.connect(lambda:self.bilateralFilterImage(self.img))
		self.buttonMedianBlur.clicked.connect(lambda:self.medianFilter(self.img))
		self.buttonwatershed.clicked.connect(lambda:self.watershedNormal(self.bilateralFilterImage(self.img)))


		self.buttonSITKblur.clicked.connect(self.blurredImage)
		self.buttonSITKgrey.clicked.connect(self.overlayBlurAndGreyMatter)
		self.buttonSITKwhite.clicked.connect(self.overlayBlurAndWhitMatter)
		self.buttonSITKfinal.clicked.connect(self.imageWhiteAndGrey)
		
		self.label = QtGui.QLabel('Get The Image')
		self.label1 = QtGui.QLabel('Normal Filters')
		self.label2= QtGui.QLabel('Combined Filters')

		#Setting the layout
		#h box's are horizontal and vbox are vertical layouts 
		hbox = QtGui.QHBoxLayout()
		
		vbox = QtGui.QVBoxLayout()

		hboxlabel=QtGui.QHBoxLayout()
		hboxlabel.addWidget(self.label)
		
		hboxInternal=QtGui.QHBoxLayout()
		hboxInternal.addWidget(self.buttonOpenDicomFile)
		hboxInternal.addWidget(self.buttonShowImageNormal)

		hboxlabel1=QtGui.QHBoxLayout()
		hboxlabel1.addWidget(self.label1)

		hboxInternal1=QtGui.QHBoxLayout()
		hboxInternal1.addWidget(self.buttonguassianThresh)
		hboxInternal1.addWidget(self.buttonGaussianBlur)
		hboxInternal1.addWidget(self.buttonBilateralFilter)
		hboxInternal1.addWidget(self.buttonMedianBlur)
		hboxInternal1.addWidget(self.buttonwatershed)

		hboxlabel2=QtGui.QHBoxLayout()
		hboxlabel2.addWidget(self.label2)

		hboxInternal2=QtGui.QHBoxLayout()
		hboxInternal2.addWidget(self.buttonSITKblur)
		hboxInternal2.addWidget(self.buttonSITKgrey)
		hboxInternal2.addWidget(self.buttonSITKwhite)
		hboxInternal2.addWidget(self.buttonSITKfinal)

		vbox.addLayout(hboxlabel)
		vbox.addLayout(hboxInternal)
		vbox.addLayout(hboxlabel1)
		vbox.addLayout(hboxInternal1)
		vbox.addLayout(hboxlabel2)
		vbox.addLayout(hboxInternal2)

		

		vbox1 = QtGui.QVBoxLayout()
		vbox1.addWidget(self.toolbar)
		vbox1.addWidget(self.canvas)

		hbox.addLayout(vbox)
		hbox.addLayout(vbox1)
		
		self.setLayout(hbox)

	def selectFile(self):
		fileName=QtGui.QFileDialog.getOpenFileName(self,'select a file:')
		return fileName

	#for showing single Dicom File
	def readSingleDicom(self,img,title=None, margin=0.05, dpi=40):
		#"MyHead/MR000000.dcm"
		filename=self.selectFile()
		ds=dicom.read_file(filename)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		ax.imshow(ds.pixel_array)
		self.canvas.draw()

	#function for opening a folder
	def selectFolder(self):
		path=QtGui.QFileDialog.getExistingDirectory(self, 'Select a folder:', '\\', QtGui.QFileDialog.ShowDirsOnly)
		return path

	#preprocessing Dicom files
	def processImage(self,PathDicom):
		#create an emplty array to list the dicom files
		listFilesDCM = []
		for dirName, subdirList, fileList in os.walk(PathDicom):
			for filename in fileList:
				if ".dcm" in filename.lower():
					listFilesDCM.append(os.path.join(dirName,filename))
		print(listFilesDCM)
		for x in range(0,len(listFilesDCM)):
			print(listFilesDCM[x])
		RefDs = dicom.read_file(listFilesDCM[0])
		ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(listFilesDCM))
		ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
		#
		x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
		y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
		z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

		ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

		for filenameDCM in listFilesDCM:
			ds = dicom.read_file(filenameDCM)
			ArrayDicom[:, :, listFilesDCM.index(filenameDCM)] = ds.pixel_array
		
		#3 kinds of image outputs.u can chose ..here we are usin only nda 1
		nda1=np.flipud(ArrayDicom[:, :, 80])				#get an array from the original image(array)
		nda2=np.flipud(ArrayDicom[:, 80, :])
		nda3=np.flipud(ArrayDicom[80, :, :])
		return x,y,nda1,nda2,nda3
	
	#showing the image after getting value from processing	
	def showImageNormal(self,title=None, margin=0.05, dpi=40):
		fig = plt.figure(dpi=dpi)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		path=self.selectFolder()
		normalX,normalY,normalZ,sided,top=self.processImage(path)
		ax.pcolormesh(normalX,normalY,normalZ,cmap=plt.get_cmap('gray'))
		#for saving the picture
		plt.pcolormesh(normalX,normalY,normalZ,cmap=plt.get_cmap('gray'))
		plt.savefig("normal1.jpg", bbox_inches="tight") 
		self.canvas.draw()


	#load normal opencv image
	def cvImage(self,title=None, margin=0.05, dpi=40):
		img=cv2.imread("normal.jpg")
		arr = np.asarray(img)
		fig=plt.figure(dpi=dpi)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		ax.imshow(img)
		self.canvas.draw()
	
	#gaussian Threshhold	
	def gaussianThreshhold(self,title=None, margin=0.05, dpi=40):
		gausThresh=cv2.adaptiveThreshold(self.grayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		ax.imshow(gausThresh)
		self.canvas.draw()
		return gausThresh
	
	#applying gaussian Filter
	def gaussianFilter(self,img,title=None, margin=0.05, dpi=40):
		kernel=np.ones((4,4),np.float32)/16
		gaussianSmooth=cv2.filter2D(img,-1,kernel)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		ax.imshow(gaussianSmooth)
		self.canvas.draw()
		return gaussianSmooth

	#bilateral filtering
	def bilateralFilterImage(self,img,title=None, margin=0.05, dpi=40):
		bilateralblur = cv2.bilateralFilter(img,10,15,15)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		ax.imshow(bilateralblur)
		self.canvas.draw()
		return bilateralblur

	#median Filter
	def medianFilter(self,img,title=None, margin=0.05, dpi=40):
		median = cv2.medianBlur(img,5)
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		ax.imshow(median)
		self.canvas.draw()
		return median
	#main output using watershed algorithm
	def watershedNormal(self,img,title=None, margin=0.05, dpi=40):
		ret, thresh = cv2.threshold(self.grayscale,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		# noise removal
		kernel = np.ones((3,3),np.uint8)
		opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
		# sure background area
		sure_bg = cv2.dilate(opening,kernel,iterations=3)
		# Finding sure foreground area
		dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
		ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
		
		# Finding unknown region
		sure_fg = np.uint8(sure_fg)
		unknown = cv2.subtract(sure_bg,sure_fg)

		# Marker labelling
		ret, markers = cv2.connectedComponents(sure_fg)
		# Add one to all labels so that sure background is not 0, but 1
		markers = markers+1

		# Now, mark the region of unknown with zero
		markers[unknown==255] = 0
		markers = cv2.watershed(img,markers)
		img[markers == -1] = [255,0,0]
		ax = self.figure.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
		ax.imshow(img)
		self.canvas.draw()


	#SimpleITK
	def initializeSITK(self):
		pathDicom =self.selectFolder()
		idxSlice = 50
		labelWhiteMatter = 1
		labelGrayMatter = 2
		reader = SimpleITK.ImageSeriesReader()						#read all the files
		filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)	#save the file names into an array
		reader.SetFileNames(filenamesDICOM)							#save the list of dicom images in an original image array
		imgOriginal = reader.Execute()
		imgOriginal = imgOriginal[:,:,idxSlice]
		imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,timeStep=0.125,numberOfIterations=5)	#function for converting smooth image array
		return labelWhiteMatter,labelGrayMatter,imgOriginal,imgSmooth


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

	#function for showing the original image on clicked original image
	def originalImage(self):
		labelWhiteMatter,labelGrayMatter,imgOriginal,imgSmooth=self.initializeSITK()
		self.sitk_show(imgOriginal)

	#function for converting blurry image
	def blurredImage(self):
		labelWhiteMatter,labelGrayMatter,imgOriginal,imgSmooth=self.initializeSITK()
		self.sitk_show(imgSmooth)

	#function for white matter
	def overlayBlurAndWhitMatter(self):	
		labelWhiteMatter,labelGrayMatter,imgOriginal,imgSmooth=self.initializeSITK()	
		lstSeeds = [(150  ,75)]
		imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,seedList=lstSeeds,lower=130,upper=190,replaceValue=labelWhiteMatter)
		imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())
		imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=labelWhiteMatter)
		self.sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatterNoHoles))

	#function for Grey Matter
	def overlayBlurAndGreyMatter(self):
		labelWhiteMatter,labelGrayMatter,imgOriginal,imgSmooth=self.initializeSITK()
		lstSeeds = [(119, 83), (198, 80), (185, 102), (164, 43)]
		imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,seedList=[(150,75)],lower=130,upper=190,replaceValue=labelWhiteMatter)
		imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())
		imgGrayMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,seedList=lstSeeds,lower=150,upper=270,replaceValue=labelGrayMatter)
		imgGrayMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGrayMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=labelGrayMatter)
		self.sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgGrayMatterNoHoles))

	#function for both white and grey matter
	def imageWhiteAndGrey(self):
		labelWhiteMatter,labelGrayMatter,imgOriginal,imgSmooth=self.initializeSITK()
		lstSeeds = [(119, 83), (198, 80), (185, 102), (164, 43)]
		imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,seedList=[(150,75)],lower=130,upper=190,replaceValue=labelWhiteMatter)
		imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())
		imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=labelWhiteMatter)
		imgGrayMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,seedList=lstSeeds,lower=150,upper=270,replaceValue=labelGrayMatter)
		imgGrayMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGrayMatter,radius=[2]*3,majorityThreshold=1,backgroundValue=0,foregroundValue=labelGrayMatter)
		imgLabels = imgWhiteMatterNoHoles | imgGrayMatterNoHoles
		self.sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgLabels))

		
#main functions for showing the window
def run():
	app=QtGui.QApplication(sys.argv)
	GUI=Window()		#shows the window class
	GUI.show()
	sys.exit(app.exec_())

run()
