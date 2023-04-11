# Class to store various data per calibration image 

class CalibImageData():
    
    # mono calibration image file path
    imageFilePath = ""
    
    
    reprojErrorArray_left = []
    reprojErrorArray_right = []

    reprojImgPoints_left  = []
    reprojImgPoints_right = []

    rawLeftImg  = []
    rawRightImg = []
    
    # Reprojection error statistics
    # min/max and standard variation per image 
    averageReprojError_left = 0
    maxReprojError_left  = 0
    minReprojError_left  = 0 
    stdReprojError_left  = 0
    
    averageReprojError_right = 0
    maxReprojError_right  = 0
    minReprojError_right  = 0 
    stdReprojError_right  = 0
    
    imageIndex = 0 
    
    isPatternFound = False
    
    # mono calibration image points 
    imagePoints = []
    
    # stereo calibraiton image points
    leftImagePoints  = []
    rightImagePoints = []
    
    # mono calibration r and t vectors 
    rvec = []
    tvec = []
    
    # stereo calibration r and t vectors
    left_rvec  = []
    left_tvec  = []
    right_rvec = []
    right_tvec = []
    
    def __init__(self):
        TBD = True
    