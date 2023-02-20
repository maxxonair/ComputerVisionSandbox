import cv2 as cv
import numpy as np

def rectifyStereoImageSet(imgl, imgr, undistMaps ):
    lmapx = undistMaps['leftUndistortionMap_x']
    lmapy = undistMaps['leftUndistortionMap_y']
    rmapx = undistMaps['rightUndistortionMap_x']
    rmapy = undistMaps['rightUndistortionMap_y']

    rimgl = cv.remap(imgl, lmapx, lmapy, cv.INTER_LANCZOS4)
    rimgr = cv.remap(imgr, rmapx, rmapy, cv.INTER_LANCZOS4)

    return (rimgl, rimgr)

def createDisparityMap(rimgl, rimgr):
    # SGBM Parameters -----------------
    # wsize default 3; 5; 7 for SGBM reduced size image; 
    # 15 for SGBM full size image (1300px and above); 
    window_size = 5

    left_matcher = cv.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
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
    lmbda = 80000
    sigma = 1.3
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


def computeDepthMapFromDispMap(dispMap, P2):
    # Depth = (focalLength * StereoBaseline) / disparity_px
    focalLength     = float(P2[0,0])
    stereoBaseLine  = float(P2[0,3]/P2[0,0])
    intDisp = dispMap.astype(float)
    intDisp[intDisp == 0] = np.NAN
    return -1 * ( focalLength * stereoBaseLine ) / intDisp