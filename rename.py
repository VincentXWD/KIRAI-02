import re
import os
import sys


prefix = 'test-segmentation-'
root_path = './submit_nii/'
ptn = re.compile('sample-(\d+)')

for rt, dirs, files in os.walk(root_path):
  for i in range(0, len(files)):
    os.rename(root_path+files[i], root_path+prefix+ptn.findall(files[i])[0]+'.nii')
