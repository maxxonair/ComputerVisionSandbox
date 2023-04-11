



# Logitech C270 webcam constants
# Native resolution 1280x960
# For now resized to 600x600 
# Focal length in meter:
# F_x_m = fx * W / w
# fx - focal length in Pixel
# W - Sensor width in Pixel 
# w - Sensor width in meter

C270_PIXEL_SIZE_M = 0.0000028

C270_SENSOR_WIDTH_M = 0.003584

C270_SENSOR_HEIGHT_M = 0.002016

C270_NATIVE_RESOLUTION_X_PX = 1280
C270_NATIVE_RESOLUTION_Y_PX = 960

# [!] Output image resolution for calibration input images
# Note: These images are always squared images
IMG_OUT_RESOLUTION_XY_PX    = 900

# Cropping stripe size in pixel 
# This assumes that width is always larger than image height
IMG_OUT_CROP_X_PX = int((C270_NATIVE_RESOLUTION_X_PX - C270_NATIVE_RESOLUTION_Y_PX) / 2 ) 