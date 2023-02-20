
import numpy as np 
import math

class StereoCalibrationData:

    K1 = []
    D1 = []
    K2 = []
    D2 = []
    R  = []
    T  = []
    E  = []
    F  = []
    
    R1 = []
    P1 = []
    R2 = []
    P2 = []
    
    Q  = []
    
    def __init__(self, log):
        self.log = log
        
    def printStereoCalibrationData(self, imgSize):
        self.log.pLogMsg("")
        self.log.pLogMsg(' [Stereo Calibration Results]')
        self.log.pLogMsg("")
        self.log.pLogMsg("[K1] "+str(self.K1))
        self.log.pLogMsg("")
        self.log.pLogMsg("[D1] "+str(self.D1))
        self.log.pLogMsg("")
        self.log.pLogMsg("[K2] "+str(self.K2))
        self.log.pLogMsg("")
        self.log.pLogMsg("[D2] "+str(self.D2))
        self.log.pLogMsg("")
        self.log.pLogMsg("[R] "+str(self.R))
        self.log.pLogMsg("")
        self.log.pLogMsg("[T] "+str(self.T))
        self.log.pLogMsg("")
        self.log.pLogMsg("[E] "+str(self.E))
        self.log.pLogMsg("")
        self.log.pLogMsg("[F] "+str(self.F))
        self.log.pLogMsg("")
        self.log.pLogMsg("[R1] "+str(self.R1))
        self.log.pLogMsg("")
        self.log.pLogMsg("[P1] "+str(self.P1))
        self.log.pLogMsg("")
        self.log.pLogMsg("[R2] "+str(self.R2))
        self.log.pLogMsg("")
        self.log.pLogMsg("[P2] "+str(self.P2))
        self.log.pLogMsg("")
        # Append summary of processed stereo bench properties:
        self.log.pLogMsg("Stereo baseline       [m]: {}".format(abs(self.P2[0,3]/self.P2[0,0])))
        self.log.pLogMsg("Calibrated FoV in x [deg]: {}".format(math.degrees(2*np.arctan(imgSize[0]/(2*self.P1[0,0])))))
        self.log.pLogMsg("Calibrated FoV in y [deg]: {}".format(math.degrees(2*np.arctan(imgSize[1]/(2*self.P1[1,1])))))