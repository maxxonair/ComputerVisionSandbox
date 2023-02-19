import cv2 as cv


def rectifyStereoImageSet(imgl, imgr, undistMaps ):
    lmapx = undistMaps['leftUndistortionMap_x']
    lmapy = undistMaps['leftUndistortionMap_y']
    rmapx = undistMaps['rightUndistortionMap_x']
    rmapy = undistMaps['rightUndistortionMap_y']

    rimgl = cv.remap(imgl, lmapx, lmapy, cv.INTER_LANCZOS4)
    rimgr = cv.remap(imgr, rmapx, rmapy, cv.INTER_LANCZOS4)

    return (rimgl, rimgr)
