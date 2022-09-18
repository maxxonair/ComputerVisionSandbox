# Class to store various data per calibration image 

class CalibImageData():
    
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
    
    imagePoints = []
    
    rvec = []
    tvec = []
    
    def __init__(self):
        TBD = True
    