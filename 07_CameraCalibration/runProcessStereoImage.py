
import cv2
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import math


# Filepath to calibration parameters
sFilePathCalibrationRoot        = "./_stereoBench_calibration_11_02_2023/2023_02_11__18_20_00_camera_01_calibration"
sFilePathCalibrationSubFolder   = "/03_camera_parameters/stereo_calibration.yaml"

sFilePathCalibrationParameters  = sFilePathCalibrationRoot + sFilePathCalibrationSubFolder
sFilePathCalibrationData        = os.path.join(sFilePathCalibrationRoot, '03_camera_parameters')

sFilePathTestImage = os.path.join('01_imageCapture/2023_35_12_Feb_02_1676237716_frame_capture/','img_0.png')

# Camera calibration parameters
K1 = []
D1 = []
K2 = []
D2 = []
P2 = []
R1 = []
R2 = []
Q  = []

# -------------------------------------------------------------------------------------------------------------
#           [Functions]
# -------------------------------------------------------------------------------------------------------------
def _loadUndistMaps():
    print()
    print('[LOAD] Undistortion maps.')
    print()
    stacked = cv2.imread(os.path.join(sFilePathCalibrationData,'caml_undistortion_map.tiff'), cv2.IMREAD_UNCHANGED)
    
    # Check file has been found and loaded before moving on 
    if stacked is None:
        print('[ERR] Left camera undistortion map not found! Exiting')
        exit(1)  
    else:
        print('Left camera undistortion map has been found and loaded.')
          
    mapx = stacked[:,:,0:2].astype(np.int16)
    mapy = stacked[:,:,2]
    (caml_mapx, caml_mapy) = cv2.convertMaps(map1=mapx, map2=mapy, dstmap1type=cv2.CV_32FC1)
    
    stackedr = cv2.imread(os.path.join(sFilePathCalibrationData,'camr_undistortion_map.tiff'), cv2.IMREAD_UNCHANGED)
    
    # Check file has been found and loaded before moving on 
    if stackedr is None:
        print('[ERR] Left camera undistortion map not found! Exiting')
        exit(1) 
    else:
        print('Right camera undistortion map has been found and loaded.')
        
    mapx = stackedr[:,:,0:2].astype(np.int16)
    mapy = stackedr[:,:,2]
    (camr_mapx, camr_mapy) = cv2.convertMaps(map1=mapx, map2=mapy, dstmap1type=cv2.CV_32FC1)
    print()
    return (caml_mapx, caml_mapy, camr_mapx, camr_mapy)

def _loadCalibrationParameters():
    global K1, D1, K2, D2, P2, R1, R2, Q
    print()
    print('[LOAD] Intrinsic camera matrices.')
    print()
    fileStorage = cv2.FileStorage()
    suc = fileStorage.open(sFilePathCalibrationParameters, cv2.FileStorage_READ)
    
    if suc:
        print('Load calibration from file.')
        
        K1 = fileStorage.getNode('K1').mat()
        D1 = fileStorage.getNode('D1').mat()
        K2 = fileStorage.getNode('K2').mat()
        D2 = fileStorage.getNode('D2').mat()
        P2 = fileStorage.getNode('P2').mat()
        R1 = fileStorage.getNode('R1').mat()
        R2 = fileStorage.getNode('R2').mat()
        Q  = fileStorage.getNode('Q').mat()

        print()
        print("K1: "+str(K1))
        print()
        print("D1: "+str(D1))
        print()
        print("K2: "+str(K2))
        print()
        print("D2: "+str(D2))
        print()
    else :
        print("Loading calibration files failed. Check file path.")
    
    fileStorage.release()
    
def _rectifyImage(img, mapx, mapy ):
    return cv2.remap(img, mapx, mapy, cv2.INTER_LANCZOS4)

def _computeDepthMap(imgL, imgR):
        # SGBM Parameters -----------------
        # wsize default 3; 5; 7 for SGBM reduced size image; 
        # 15 for SGBM full size image (1300px and above); 
        window_size = 5

        left_matcher = cv2.StereoSGBM_create(
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
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        # left_matcher = cv2.StereoBM_create()
        window_size = 15
        min_disp = 16
        num_disp = 96 - min_disp
        left_matcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                                    numDisparities=num_disp,
                                    blockSize=16,
                                    P1=8 * 3 * window_size ** 2,
                                    P2=32 * 3 * window_size ** 2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=150,
                                    speckleRange=32
                                    )
        disp = left_matcher.compute(imgL, imgR).astype(float)
        # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # # FILTER Parameters
        # lmbda = 80000
        # sigma = 1.3
        # visual_multiplier = 6

        # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        # wls_filter.setLambda(lmbda)

        # wls_filter.setSigmaColor(sigma)
        # displ = left_matcher.compute(imgL, imgR)  
        # dispr = right_matcher.compute(imgR, imgL)  
        # displ = np.int16(displ)
        # dispr = np.int16(dispr)
        # filteredImg = wls_filter.filter(displ, imgL, None, dispr)  

        # filteredImg = cv2.normalize(src=filteredImg, 
        #                            dst=filteredImg, 
        #                            beta=0, alpha=255, 
        #                            norm_type=cv2.NORM_MINMAX)
        
        # filteredImg = np.uint8(filteredImg)

        return disp
    
def _loadStereoImage():
    img = cv2.imread(sFilePathTestImage)
    # Convert image to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h, w) = gray.shape[:2]
    # Split dual image into left&right image
    # + -----> w/x
    # | [][][][][][][][][][][][][][][]
    # | [][][][][][][][][][][][][][][]
    # | [][][][][][][][][][][][][][][]
    # | [][][][][][][][][][][][][][][]
    # V
    # h/y             | w2
    #    left image  < > right image
    # 
    # Compute separator x position
    w2 = int(w/2)
    # Extract left & right image
    leftImage  = gray[ 0:h , 0:w2 ]
    rightImage = gray[ 0:h , w2:w ]
    return (leftImage, rightImage)
# -------------------------------------------------------------------------------------------------------------
#                   [Process Sequence]
# -------------------------------------------------------------------------------------------------------------
# Load calibration matrices
_loadCalibrationParameters()

# Load undistortion maps 
(camr_mapx, camr_mapy, caml_mapx, caml_mapy) = _loadUndistMaps()

# Load test image 
( rightImage, leftImage) = _loadStereoImage()

# Undistort image 
print('Rectify left image')
imgl_rect = _rectifyImage(leftImage, caml_mapx, caml_mapy)
print('Rectify right image')
imgr_rect = _rectifyImage(rightImage, camr_mapx, camr_mapy)
print()

# Compute disparity map 
print('Compute disparity map.')
disp_map = _computeDepthMap(imgl_rect, imgr_rect)
disp_map = np.asarray(disp_map)


disparityF = disp_map.astype(float)
maxv = np.max(disparityF.flatten())
minv = np.min(disparityF.flatten())
disparityF = 255.0*(disparityF-minv)/(maxv-minv)
disparityU = disparityF.astype(np.uint8)

print('Compute point cloud')
im_3D = cv2.reprojectImageTo3D(disparityU, Q)

# Compute depth map
(h, w , d) = im_3D.shape
depth_map = np.zeros([h,w], dtype=float)

for iCounter in range(h):
    for jCounter in range(w):
        depth_map[iCounter, jCounter] = math.sqrt( im_3D[iCounter, jCounter, 0] * im_3D[iCounter, jCounter, 0] 
                                                 + im_3D[iCounter, jCounter, 1] * im_3D[iCounter, jCounter, 1] 
                                                 + im_3D[iCounter, jCounter, 2] * im_3D[iCounter, jCounter, 2] )

# Plot map 
# fig = plt.figure(figsize=(6, 3.2))
titleFontSize = 8
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

im = ax1.imshow(imgl_rect, cmap='gray')
ax1.set_title('Left Image', fontsize=titleFontSize)
ax1.set_aspect('equal')

im = ax2.imshow(imgr_rect, cmap='gray')
ax2.set_title('Right Image', fontsize=titleFontSize)
ax2.set_aspect('equal')

im = ax3.imshow(disparityU, cmap='viridis')
ax3.set_title('Disparity Map', fontsize=titleFontSize)
ax3.set_aspect('equal')
cb = plt.colorbar(im)

im2 = ax4.imshow(depth_map, cmap='viridis')
ax4.set_title('Depth Map', fontsize=titleFontSize)
ax4.set_aspect('equal')
cb = plt.colorbar(im2)

plt.tight_layout()
# Save image to file
plt.savefig('test.png', dpi=400)
# -------------------------------------------------------------------------------------------------------------



