#_*_ coding: utf: 8 _*_
import SimpleITK as sitk
from myshow import *
import math
from my_logging import *


def get_predicted_liver_label(pred_image_name: str) -> sitk.Image:
  image = sitk.ReadImage(pred_image_name, sitk.sitkUInt8)
  size = image.GetSize()
  output = sitk.Image(size, sitk.sitkUInt8)
  output.CopyInformation(image)
  for i in range(0, size[0]):
    for j in range(0, size[1]):
      if image.GetPixel(i, j) == 128 or image.GetPixel(i, j) == 255:
        output.SetPixel(i, j, 1)
  return output


def get_raw_liver_image(raw: sitk.Image, liver: sitk.Image) -> sitk.Image:
  size = raw.GetSize()
  output = sitk.Image(size, sitk.sitkUInt8)
  output.CopyInformation(raw)
  for i in range(0, size[0]):
    for j in range(0, size[1]):
      if liver.GetPixel(i, j) == 1:
        output.SetPixel(i, j, raw.GetPixel(i, j))
  return output


def get_tumor_seed(pred_image_name: str) -> list:
  image = sitk.ReadImage(pred_image_name, sitk.sitkUInt8)
  assert isinstance(image, sitk.Image)
  size = image.GetSize()
  seed = []
  vis = [[0 for _ in range(0, size[0])] for _ in range(0, size[1])]

  def bfs(x: int, y: int) -> list:
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
  return seed


def level_set_cut_v1(image: sitk.Image, seed: list, pred_image_name: str) -> (sitk.Image, sitk.Image):
  assert isinstance(image, sitk.Image) and image.GetPixelID() == sitk.sitkUInt8

  seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
  seg.CopyInformation(image)

  seed = map(lambda each: list(map(int, each)), seed)
  for each in seed:
    seg[each] = 1
  logger.info(pred_image_name+' have '+str(len(list(seed)))+' lesion region(s)')

  seg = sitk.BinaryDilate(seg, 3)
  assert isinstance(seg, sitk.Image)

  # myshow(sitk.LabelOverlay(image, seg), title="Initial Seed")

  stats = sitk.LabelStatisticsImageFilter()
  stats.Execute(image, seg)

  factor = 1.8
  lower_threshold = stats.GetMean(1) - factor * stats.GetSigma(1) # - math.log(stats.GetMean(1))
  upper_threshold = stats.GetMean(1) + factor * stats.GetSigma(1) # + math.log(stats.GetMean(1))
  logger.info('the lower_threshold and upper_threshold :'+ \
              str(lower_threshold)+' '+str(upper_threshold))
  if lower_threshold == 0 or upper_threshold == 0:
    logger.warn('Threshold Error. Ignoring...')
    return image, image, -1

  init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)
  lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
  lsFilter.SetLowerThreshold(lower_threshold)
  lsFilter.SetUpperThreshold(upper_threshold)
  lsFilter.SetMaximumRMSError(0.02)
  lsFilter.SetNumberOfIterations(500)
  lsFilter.SetCurvatureScaling(.5)
  lsFilter.SetPropagationScaling(1)
  lsFilter.ReverseExpansionDirectionOn()
  ls = lsFilter.Execute(init_ls, sitk.Cast(image, sitk.sitkFloat32))

  assert isinstance(ls, sitk.Image)
  return sitk.LabelOverlay(image, ls>0), ls, 1


def relabelling(image: sitk.Image) -> sitk.Image:
  relabeler = sitk.RelabelComponentImageFilter()
  ret = relabeler.Execute(image)
  assert isinstance(ret, sitk.Image)

  return image - ret


def gradient_filting(image: sitk.Image) -> sitk.Image:
  gradient_filter = sitk.GradientAnisotropicDiffusionImageFilter()
  gradient_filter.SetNumberOfIterations(20) # default: 15
  gradient_filter.SetTimeStep(0.0625)
  gradient_filter.SetConductanceParameter(3.0)
  return gradient_filter.Execute(image)


def padding_process(raw_image_name: str, pred_image_name: str, save_path: str) -> None:
  '''
  anisotropic diffusing ->
  relabel filting ->
  hmax contrasting ->
  level cut thresholding ->
  Hole Filling .
  :return: None
  '''
  seed = get_tumor_seed(pred_image_name)
  if len(seed) == 0:
    logger.warn('This image does not have tumor pixels.')
    sitk.WriteImage(sitk.ReadImage(pred_image_name, sitk.sitkUInt8), save_path)
    return

  raw_img = sitk.ReadImage(raw_image_name, sitk.sitkUInt8)
  assert isinstance(raw_img, sitk.Image)
  liver_label = get_predicted_liver_label(pred_image_name)
  raw = get_raw_liver_image(raw_img, liver_label)
  raw = sitk.Cast(raw, sitk.sitkFloat32)

  # (a) Anisotropic diffusion
  gradient = gradient_filting(raw)

  # (b) Hmaxima high contrasting
  # Relabel & Hmaxima
  relabel = relabelling(sitk.Cast(gradient, sitk.sitkUInt8))
  hmax_filter = sitk.HMaximaImageFilter()
  hmax = hmax_filter.Execute(relabel)
  # (c) Level-set thresholding
  _, label, signal = level_set_cut_v1(sitk.Cast(hmax, sitk.sitkUInt8), seed, pred_image_name)
  if signal == -1:
    sitk.WriteImage(sitk.ReadImage(pred_image_name, sitk.sitkUInt8), save_path)
    return

  # (d) Hole filling
  hole_filter = sitk.GrayscaleFillholeImageFilter()
  label = hole_filter.Execute(label)
  myshow(_)
  myshow(sitk.ReadImage('./images/mask_png/sample-117/sample-117-slice-470.png'))

def run():
  raw_image_name = './images/raw_jpg/sample-117/sample-117-slice-470.jpg'
  pred_image_name = './images/pred_png/sample-117/sample-117-slice-470.png'
  save_path = './sample-117-slice-470.png'
  padding_process(raw_image_name, pred_image_name, save_path)


if __name__ == '__main__':
  run()
