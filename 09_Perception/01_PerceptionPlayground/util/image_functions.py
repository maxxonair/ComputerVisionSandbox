'''

@description
This file contains a set of useful interface functions to make the best use of 
OpenCV and:
- Undistort/rectify raw camera images
- Create disparity maps
- Create camera calibration


@dependencies: This class requires OpenCV

Copyright © 2024 Max Braun

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the “Software”), to deal in the 
Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, subject 
to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import cv2 as cv
import numpy as np
#-------------------------------------------------------------------------------
#                       [stereo camera calibration]
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#                       [stereo rectification]
#-------------------------------------------------------------------------------
def rectify_stereo_image(imgl, imgr, undistMaps ):
    '''
    Rectify stereo image frame using a given set of undistortion maps (stored in
    a dictionary, format see below)
    
    '''
    lmapx = undistMaps['leftUndistortionMap_x']
    lmapy = undistMaps['leftUndistortionMap_y']
    rmapx = undistMaps['rightUndistortionMap_x']
    rmapy = undistMaps['rightUndistortionMap_y']

    rimgl = cv.remap(imgl, lmapx, lmapy, cv.INTER_LANCZOS4)
    rimgr = cv.remap(imgr, rmapx, rmapy, cv.INTER_LANCZOS4)

    return (rimgl, rimgr)

#-------------------------------------------------------------------------------
#                       [disparity maps]
#-------------------------------------------------------------------------------
def create_disparity_map(rimgl, rimgr):
    '''
    Create disparity map from stereo image (input: left and right image)
    '''
    # SGBM Parameters -----------------
    # wsize default 3; 5; 7 for SGBM reduced size image; 
    # 15 for SGBM full size image (1300px and above); 
    window_size = 5

    left_matcher = cv.StereoSGBM_create(
        minDisparity=-1,
        # max_disp has to be dividable by 16 f. E. HH 192, 256
        numDisparities=5*16,  
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full 
        # size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 15000 # 80000
    sigma =  1.3
    visual_multiplier = 6

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(rimgl, rimgr)  
    dispr = right_matcher.compute(rimgr, rimgl)  
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, rimgl, None, dispr)  

    filteredImgNormalized = cv.normalize(src=filteredImg, 
                                dst=filteredImg, 
                                beta=0, alpha=255, 
                                norm_type=cv.NORM_MINMAX)
    
    filteredImgNormalized = np.uint8(filteredImgNormalized)

    return (displ, filteredImg, filteredImgNormalized)

def compute_depth_map_from_disp_map(dispMap, P2):
    '''
    Compute depth map (distance object to camera center for each pixel) from 
    disparity map (disparity in x between left and right image for each pixel).
    
    '''
    # Formula: Depth = (focalLength * StereoBaseline) / disparity_px
    focalLength     = float(P2[0,0])
    stereoBaseLine  = float(P2[0,3]/P2[0,0])
    intDisp = dispMap.astype(float)
    intDisp[intDisp == 0] = np.NAN
    return -1 * ( focalLength * stereoBaseLine ) / intDisp
#-------------------------------------------------------------------------------