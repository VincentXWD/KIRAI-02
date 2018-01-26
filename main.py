#-*- coding: utf-8 -*-
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from myshow import *
import multiprocessing


# ImgName = './hakase.jpg'
raw_image_name =   './images/raw_jpg/sample-27/sample-27-slice-389.jpg'
mask_image_name = './images/mask_png/sample-27/sample-27-slice-389.png'
pred_image_name = './images/pred_png/sample-27/sample-27-slice-389.png'


def CheckPixel(p1, p2):
  return p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]


def GetLiverSeed(pred_png_name):
  pred_image = cv2.imread(pred_image_name)
  seed = [0, 0]
  tot = 0
  # seed = []
  for i in range(0, 512):
    for j in range(0, 512):
      if CheckPixel(pred_image[i, j], [128, 128, 128]):
        # seed.append([j, i])
        seed[0] += j
        seed[1] += i
        tot += 1
  seed[0] = float(seed[0]) / float(tot)
  seed[1] = float(seed[1]) / float(tot)
  return seed


def GetTumorSeed(pred_png_name):
  pred_image = cv2.imread(pred_image_name)
  seed = []
  for i in range(0, 512):
    for j in range(0, 512):
      if CheckPixel(pred_image[i, j], [255, 255, 255]):
        seed.append([j, i])
  if len(seed) == 0:
    print('Warning: this image does not have tumor pixels.')
    return -1
  return seed


def LevelSetCut(image, seed):
  assert isinstance(image, sitk.Image) and image.GetPixelID() == sitk.sitkUInt8

  # 各向异性滤波处理
  image = sitk.GradientAnisotropicDiffusion(sitk.Cast(image, sitk.sitkFloat32))
  image = sitk.Cast(image, sitk.sitkUInt8)

  seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
  seg.CopyInformation(image)
  seg = sitk.BinaryDilate(seg, 3)
  assert isinstance(seg, sitk.Image)

  seed = map(int, seed)
  seg[seed] = 1

  # seed = map(lambda each: map(int, each), seed)
  # for each in seed:
  #   print each
  #   seg[each] = 1

  myshow(sitk.LabelOverlay(image, seg), title="Initial Seed")

  # Use the seed to estimate a reasonable threshold range.
  stats = sitk.LabelStatisticsImageFilter()
  stats.Execute(image, seg)

  factor = 3.5
  lower_threshold = stats.GetMean(1) - factor * stats.GetSigma(1)
  upper_threshold = stats.GetMean(1) + factor * stats.GetSigma(1)
  print('the lower_threshold and upper_threshold :', lower_threshold, upper_threshold)

  init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)
  lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
  lsFilter.SetLowerThreshold(lower_threshold)
  lsFilter.SetUpperThreshold(upper_threshold)
  lsFilter.SetMaximumRMSError(0.02)
  lsFilter.SetNumberOfIterations(1000)
  lsFilter.SetCurvatureScaling(.5)
  lsFilter.SetPropagationScaling(1)
  lsFilter.ReverseExpansionDirectionOn()
  ls = lsFilter.Execute(init_ls, sitk.Cast(image, sitk.sitkFloat32))

  assert isinstance(ls, sitk.Image)
  print(ls.GetPixelIDTypeAsString())
  print(ls.GetDimension())
  # myshow(sitk.LabelOverlay(image, sitk.LabelContour(ls>0), 1.0))
  myshow(sitk.LabelOverlay(image, ls>0), 'Level Set Segmentation')
  # return sitk.LabelOverlay(image, ls>0)


def ShowMaskImage(mask_image_name):
  x = sitk.ReadImage(mask_image_name, sitk.sitkUInt8)
  myshow(x)


def ProcessLevelSetCut(pred_image_name):
  global raw_image_name
  raw_image = sitk.ReadImage(raw_image_name, sitk.sitkUInt8)

  seed = GetLiverSeed(pred_image_name)
  # print 'the seed: ', seed
  img = LevelSetCut(raw_image, seed)


CONFIDENCE_CONNECTED = 0
CONNECTED_THRESHOLD = 1
def RegionGrowing(image):
  assert isinstance(image, sitk.Image) and image.GetPixelID() == sitk.sitkUInt8
  # 各向异性滤波处理
  gradient_filter = sitk.GradientAnisotropicDiffusionImageFilter()
  gradient_filter.SetNumberOfIterations(30) # default: 15
  gradient_filter.SetTimeStep(0.0625)
  gradient_filter.SetConductanceParameter(3.0)
  image = gradient_filter.Execute(image)
  image = sitk.Cast(image, sitk.sitkUInt8)

  seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
  seg.CopyInformation(image)

  seed = map(lambda each: map(int, each), GetLiverSeed(pred_image_name))
  for each in seed:
    seg[each] = 1

  # seed = map(int, GetLiverSeed(pred_image_name))
  # seg[seed] = 1
  # seg = sitk.BinaryDilate(seg, 3)

  myshow(sitk.LabelOverlay(image, seg), title="Initial Seed")

  loif = sitk.LabelOverlayImageFilter()

  if CONNECTED_THRESHOLD:
    # Connected Threshold Filter
    seg_con = sitk.ConnectedThreshold(image, seedList=seed, lower=100, upper=190)
    # seg_con = sitk.ConnectedThreshold(image, seedList=[seed], lower=100, upper=190)
    myshow(loif.Execute(image, seg_con), title="Connected Threshold")

  if CONFIDENCE_CONNECTED:
    # Confidence Connected Filter
    seg_conf = sitk.ConfidenceConnected(image, seedList=seed, numberOfIterations=1, multiplier=2.5, initialNeighborhoodRadius=1, replaceValue=1)
    # seg_conf = sitk.ConfidenceConnected(image, seedList=[seed], numberOfIterations=1, multiplier=2.5, initialNeighborhoodRadius=1, replaceValue=1)
    myshow(loif.Execute(image, seg_conf), title='Confidence Threshold')

  mask = sitk.ReadImage(mask_image_name)
  myshow(loif.Execute(image, mask), title='Mask')


def test(image):
  assert isinstance(image, sitk.Image) and image.GetPixelID() == sitk.sitkUInt8
  size = image.GetSize()
  # 各向异性滤波处理
  image = sitk.GradientAnisotropicDiffusion(sitk.Cast(image, sitk.sitkFloat32))
  image = sitk.Cast(image, sitk.sitkUInt8)
  seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
  seg.CopyInformation(image)

  # seed = map(lambda each: map(int, each), GetLiverSeed(pred_image_name))
  # for each in seed:
  #   seg[each] = 1
  seed = map(int, GetLiverSeed(pred_image_name))
  seg[seed] = 1
  seg = sitk.BinaryDilate(seg, 3)







if __name__ == '__main__':
  raw_image = sitk.ReadImage(raw_image_name, sitk.sitkUInt8)
  assert isinstance(raw_image, sitk.Image)

  # ProcessLevelSetCut(pred_image_name)
  # p1 = multiprocessing.Process(target=ShowMaskImage, args=(mask_image_name, ))
  # p1.start()
  # p2 = multiprocessing.Process(target=ProcessLevelSetCut, args=(pred_image_name, ))
  # p2.start()
  # RegionGrowing(raw_image)

  test(raw_image)