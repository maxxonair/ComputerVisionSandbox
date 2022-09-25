# Class to store various data per calibration image 

class CalibImageData():
    
    # mono calibration image file path
    imageFilePath = ""
    
    averageReprojError = 0
    
    reprojErrorArray = []
    
    # Reprojection error statistics
    # min/max and standard variation per image 
    maxReprojError = 0
    minReprojError = 0 
    stdReprojError = 0
    
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
    