import math
import cv2 as cv

class SunDetector():
  
  MAX_SUN_DETECT_ATTEMPTS = 5
  
  MIN_THR_CIRCLE = 0.7
  MAX_THR_CIRCLE = 1.2
  
  MIN_THR_RADIUS =  5
  MAX_THR_RADIUS = 25
  
  debugImg  = []
  
  def detect(self, rimgl, log):
    """
    _summary_ Compute sun center in the left image plane (x,y)
    
    """
    binaryThrStart = 254
    binaryThr      = binaryThrStart
    isSunDetected  = False
    attemptCount   = 0
    
    coordinates = [-1,-1]
    
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
      
      cv.imwrite(f'./output/binaryDebug_{attemptCount}.png', binaryImg)
      
      # find contours in the binary image
      contours, hierarchy = cv.findContours(binaryImg,
                                            cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)
      
      self.debugImg = rimgl
      circle_contours = []
      
      validCentCounter = 0
      
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
          continue
        
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        x, y, w, h = cv.boundingRect(contour)
        
        
        radius = int((w + h) / 2)

        if perimeter == 0:
          continue
        
        circularity = 4*math.pi*(area/(perimeter*perimeter))

        if (self.MIN_THR_CIRCLE < circularity < self.MAX_THR_CIRCLE
           and self.MIN_THR_RADIUS < radius < self.MAX_THR_RADIUS):
          circle_contours.append(contour)
          validCentCounter = validCentCounter + 1

          log.pLogMsg(f'Coords: {(x,y)}')
          cv.circle(self.debugImg, (x, y), radius, (235, 235, 0), -1)
          cv.putText(self.debugImg, 
                    f"centroid {validCentCounter}", 
                    (x-25, y-25),
                    cv.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2)
 
      log.pLogMsg(f'{len(circle_contours)} valid circles found')
      
      if len(circle_contours):
        isSunDetected = True
        break
      
      # Reduce binary threshold 
      binaryThr = binaryThr - 1
      # Increment attempty counter 
      attemptCount = attemptCount + 1
    
    if not isSunDetected :
      log.pLogMsg(f'Detection failed.')
    else:
      log.pLogMsg(f'Detection finished.')
      
    return isSunDetected, coordinates
    