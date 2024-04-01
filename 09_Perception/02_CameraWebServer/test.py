import cv2


try:
  img = cv2.imread('test.png')
except:
  print('failed to load image')