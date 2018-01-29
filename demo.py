#_*_ coding: utf: 8 _*_
from sys import argv
import SimpleITK as sitk
from myshow import *
import math

raw_image_name  =  './images/raw_jpg/sample-102/sample-102-slice-512.jpg'
mask_image_name = './images/mask_png/sample-102/sample-102-slice-512.png'
pred_image_name = './images/pred_png/sample-102/sample-102-slice-512.png'
save_path =                                  './sample-102-slice-512.png'


def get_predicted_liver_label():
  image = sitk.ReadImage(pred_image_name, sitk.sitkUInt8)
  assert isinstance(image, sitk.Image)
  size = image.GetSize()
  output = sitk.Image(size, sitk.sitkUInt8)
  output.CopyInformation(image)
  for i in range(0, size[0]):
    for j in range(0, size[1]):
      if image.GetPixel(i, j) == 128 or image.GetPixel(i, j) == 255:
        output.SetPixel(i, j, 1)
  return output


def get_raw_liver_image(raw, liver):
  assert raw.GetSize() == liver.GetSize()
  size = raw.GetSize()
  output = sitk.Image(size, sitk.sitkUInt8)
  output.CopyInformation(raw)
  for i in range(0, size[0]):
    for j in range(0, size[1]):
      if liver.GetPixel(i, j) == 1:
        output.SetPixel(i, j, raw.GetPixel(i, j))
  return output


def get_tumor_seed():
  image = sitk.ReadImage(pred_image_name, sitk.sitkUInt8)
  assert isinstance(image, sitk.Image)
  size = image.GetSize()
  seed = []
  vis = [[0 for _ in range(0, size[0])] for _ in range(0, size[1])]

  def bfs(x, y):
    dx, dy = [1,-1,0,0], [0,0,1,-1]
    q = [[x, y]]
    ret = [[x, y]]
    while len(q) != 0:
      px, py = q[0][0], q[0][1]
      q.pop(0)
      for i in range(0, 4):
        tx, ty = px + dx[i], py + dy[i]
        if vis[tx][ty] or \
           tx < 0 or tx >= size[0] or ty < 0 or ty >= size[1] or \
           image.GetPixel(tx, ty) != 255:
          continue
        vis[tx][ty] = 1
        ret.append([tx, ty])
        q.append([tx, ty])
    print(len(ret))
    return ret

  for i in range(0, size[0]):
    for j in range(0, size[1]):
      if image.GetPixel(i, j) == 255:
        if vis[i][j] == 1:
          continue
        ret = bfs(i, j)
        tx, ty = 0, 0
        for k in ret:
          tx += k[0]
          ty += k[1]
        tx /= len(ret)
        ty /= len(ret)
        seed.append([tx, ty])
  print(len(seed))
  return seed


def level_set_cut(image, seed):
  assert isinstance(image, sitk.Image) and image.GetPixelID() == sitk.sitkUInt8

  seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
  seg.CopyInformation(image)

  seed = map(lambda each: list(map(int, each)), seed)
  for each in seed:
    print(each)
    seg[each] = 1

  seg = sitk.BinaryDilate(seg, 5)
  assert isinstance(seg, sitk.Image)

  # myshow(sitk.LabelOverlay(image, seg), title="Initial Seed")

  # Use the seed to estimate a reasonable threshold range.
  stats = sitk.LabelStatisticsImageFilter()
  stats.Execute(image, seg)

  factor = 1.8
  lower_threshold = stats.GetMean(1) - factor * stats.GetSigma(1) - math.log(stats.GetMean(1))
  upper_threshold = stats.GetMean(1) + factor * stats.GetSigma(1) + math.log(stats.GetMean(1))
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
  # myshow(sitk.LabelOverlay(image, ls>0), 'Level Set Segmentation')
  # myshow(sitk.LabelOverlay(image, sitk.LabelContour(ls>0), 1.0))
  return sitk.LabelOverlay(image, sitk.LabelContour(ls>0), 1.0), ls


def relabelling(image):
  assert isinstance(image, sitk.Image)
  relabeler = sitk.RelabelComponentImageFilter()
  return image - relabeler.Execute(image)


def demo1():
  '''
  anisotropic diffusing ->
  relabel filting ->
  hmax contrasting ->
  level cut thresholding .
  :return: None
  '''
  raw_img = sitk.ReadImage(raw_image_name, sitk.sitkUInt8)
  assert isinstance(raw_img, sitk.Image)
  liver_label = get_predicted_liver_label()
  raw = get_raw_liver_image(raw_img, liver_label)
  raw = sitk.Cast(raw, sitk.sitkFloat32)

  # (a) Anisotropic diffusion
  gradient_filter = sitk.GradientAnisotropicDiffusionImageFilter()
  gradient_filter.SetNumberOfIterations(20) # default: 15
  gradient_filter.SetTimeStep(0.0625)
  gradient_filter.SetConductanceParameter(3.0)
  gradient = gradient_filter.Execute(raw)

  # 增加了relabel滤波
  gradient = relabelling(sitk.Cast(gradient, sitk.sitkUInt8))

  # (b) High contrasting
  hmax_filter = sitk.HMaximaImageFilter()
  hmax = hmax_filter.Execute(gradient)

  seed = get_tumor_seed()
  if len(seed) == 0:
    print('This image does not have tumor pixels.')
    return

  # (c) Thresholding
  overlay, label = level_set_cut(sitk.Cast(hmax, sitk.sitkUInt8), seed)
  sitk.WriteImage(overlay, save_path)
  myshow(sitk.ReadImage(mask_image_name))


if __name__ == '__main__':
  demo1()
