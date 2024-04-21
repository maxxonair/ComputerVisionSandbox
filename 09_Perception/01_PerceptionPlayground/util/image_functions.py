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
    displ = left_matcher.compute(rimgl, rimgr) 
    filteredImg = displ 
    filteredImgNormalized = displ
    # right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # # FILTER Parameters
    # lmbda = 15000 # 80000
    # sigma =  1.3
    # visual_multiplier = 6

    # wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    # wls_filter.setLambda(lmbda)

    # wls_filter.setSigmaColor(sigma)
    # dispr = right_matcher.compute(rimgr, rimgl)  
    # displ = np.int16(displ)
    # dispr = np.int16(dispr)
    # filteredImg = wls_filter.filter(displ, rimgl, None, dispr)  

    # filteredImgNormalized = cv.normalize(src=filteredImg, 
    #                             dst=filteredImg, 
    #                             beta=0, alpha=255, 
    #                             norm_type=cv.NORM_MINMAX)
    
    # filteredImgNormalized = np.uint8(filteredImgNormalized)

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

#-------------------------------------------------------------------------------
#                       [stereo correlation]
#-------------------------------------------------------------------------------
def triangulateStereoFeature(p1, v1, p2, v2):
    """
    This function is to triangulate a feaure in 3D space from a stereo 
    image observation. 
            
    Args:
        p1 (np.array): Position of the first (left) camera frame in the reference
                       frame.
        v1 (np.array): Direction vector from the frist camera towards the 
                       feature in the reference frame
        p2 (np.array): Position of the first (right) camera frame in the reference
                       frame.
        v2 (np.array): Direction vector from the second camera towards the 
                       feature in the reference frame
                       
    Returns:
        featPos (np.array): Position of the feature in the reference frame
        objDistance_m (float32): Distance between the feature and the first (left)
                                 camera
                                 
    """
    # Make sure all inputs are np.array
    p1 = np.asarray(p1, np.float32)
    v1 = np.asarray(v1, np.float32)
    p2 = np.asarray(p2, np.float32)
    v2 = np.asarray(v2, np.float32)
    
    featPos_ref = np.zeros((3,1), np.float32)
    objDistance_m = 0 
    
    # Compute norm of both direction vectors
    v1Norm = np.linalg.norm(v1)
    v2Norm = np.linalg.norm(v2)
    
    # Check both direction vectors have valid length
    if v1Norm == 0:
        print(f'Length of first direction vector (v1) is zero. Exiting')
        return featPos_ref, objDistance_m
    elif v2Norm == 0:
        print(f'Length of first direction vector (v1) is zero. Exiting')
        return featPos_ref, objDistance_m
    
    # Normalize direction vectors
    v1 = v1 / v1Norm
    v2 = v2 / v2Norm
    
    # Populate A matrix and b vector 
    bVec = np.asarray([(p2[0]-p1[0]),
                       (p2[1]-p1[1]),
                       (p2[2]-p1[2])], np.float32)
    Amat = np.zeros((3,3), dtype=np.float32)
    for index in range(2):
        Amat[index,0] =  v1[index]
        Amat[index,1] = -v2[index]
    
    # Solve linear equation system using least-squares
    scaleVector = np.linalg.lstsq(Amat, bVec, rcond=None)[0]
    
    # Compile results
    featPos_ref = p1 + scaleVector[0] * v1
    objDistance_m = (scaleVector[0]).item()
    
    return featPos_ref, objDistance_m

def correlatePixel():
    toDo = True
    
#-------------------------------------------------------------------------------
#                            [misc]
#-------------------------------------------------------------------------------

def getFeatureDirectionVector(featCoordinates, intrKmat):
    """
    Compute the direction vector pointing towards a feature with the corresponding
    pixel coordinates featCoordinates in the image plane. The direction vector is 
    defined in the camera frame.

    Args:
        featCoordinates (tupel): Pixel coordinates of the the feature in the 
                                 camera image
        intrKmat (np.array): Intrinsic camera matrix
        
    Returns:
        featVec_cam (np.array): Feature direction vector in the camera frame.
    """
    
    featVec_cam = np.zeros((3,1), np.float32)
    
    # Check that focal lengths are not zero
    if intrKmat[0,0] == 0 or intrKmat[1,1] == 0:
        print("[ERROR] Focal length equal to zero. Exiting!")
        return featVec_cam
    
    featVec_cam[0] = (featCoordinates[0] - intrKmat[0,2]) / intrKmat[0,0]
    featVec_cam[1] = (featCoordinates[1] - intrKmat[1,2]) / intrKmat[1,1]
    featVec_cam[2] = 1
    
    featVec_cam = featVec_cam / np.linalg.norm(featVec_cam)
    
    return featVec_cam
    