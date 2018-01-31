from matplotlib import pyplot as plt
import cv2

img_path = './report/new/'
save_path = './report/new/'
png_names = [
  ['sample-37-slice-94.png',
   'sample-37-slice-94-ls.png',
   'sample-37-slice-94-mask.png'],
  ['sample-117-slice-350.png',
   'sample-117-slice-350-ls.png',
   'sample-117-slice-350-mask.png'],
  ['sample-117-slice-375.png',
   'sample-117-slice-375-ls.png',
   'sample-117-slice-375-mask.png'],
  ['sample-117-slice-470.png',
   'sample-117-slice-470-ls.png',
   'sample-117-slice-470-mask.png']
]


for each in png_names:
  png = []
  for pic in each:
    png.append(cv2.imread(img_path+pic))
  token = len(each) * 10 + 100
  for i in range(len(png)):
    plt.subplot(token+i+1)
    plt.imshow(png[i])
  plt.title(each[0])
  plt.show()
