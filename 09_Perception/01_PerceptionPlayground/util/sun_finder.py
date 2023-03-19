import math
import cv2 as cv

class SunDetector():
  
  MAX_SUN_DETECT_ATTEMPTS = 25
  
  MIN_THR_CIRCLE = 0.7
  MAX_THR_CIRCLE = 1.2
  
  binaryImg = []
  debugImg  = []
  
  def detect(self, rimgl, log):
    """
    _summary_ Compute sun center in the left image plane (x,y)
    
    """
    binaryThrStart = 254
    binaryThr      = binaryThrStart
    isSunDetected  = False
    attemptCount   = 0
    
    try:
      grayImg = cv.cvtColor(rimgl, cv.COLOR_BGR2GRAY)
    except:
      grayImg = rimgl
    
    while (not isSunDetected and attemptCount < self.MAX_SUN_DETECT_ATTEMPTS):
      log.pLogMsg(f' Binary Threshold: {binaryThr}')
      thr, binaryImg = cv.threshold(grayImg, 
                                    binaryThr, 
                                    255, 
                                    cv.THRESH_BINARY)
      
      # find contours in the binary image
      contours, hierarchy = cv.findContours(binaryImg,
                                                 cv.RETR_TREE,
                                                 cv.CHAIN_APPROX_SIMPLE)
      print(f'{len(contours)} contours found')
      
      # circles = cv.HoughCircles(binaryImg, 
      #                           cv.HOUGH_GRADIENT, 
      #                           2, 
      #                           minDist=30, 
      #                           param1=200, 
      #                           param2=40,
      #                           minRadius=10, 
      #                           maxRadius=20)
      
      self.debugImg = grayImg
      circle_contours = []
      
      # Filter for circular shapes in the right size 
      for contour in contours:
        # calculate moments for each contour
        M = cv.moments(contour)

        shapeInvalid = False
        # calculate x,y coordinate of center
        if M["m00"] != 0:
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])
        else:
          break
        
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))

        if self.MIN_THR_CIRCLE < circularity < self.MAX_THR_CIRCLE:
            circle_contours.append(contour)
            
        log.pLogMsg(f'{len(contours)} contours found')
        cv.circle(self.debugImg, (cX, cY), 5, (255, 255, 255), -1)
        cv.putText(self.debugImg, 
                   "centroid", 
                   (cX - 25, cY - 25),
                   cv.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (255, 255, 255), 
                   2)
 
      
      # Reduce binary threshold 
      binaryThr = binaryThr - 1
      # Increment attempty counter 
      attemptCount = attemptCount + 1
    
    if attemptCount > self.MAX_SUN_DETECT_ATTEMPTS :
      log.pLogMsg(f'Detection failed.')
    else:
      log.pLogMsg(f'Detection finished.')
    