import os 
import cv2 as cv

# Settings
CONVERT_TO_GREY = True
scale_percent = 22

# File paths
inputFolderPath = './01_ImagesToScale/'
outputFolderPath = './02_ScaledImages/'

#imageFile = 'IMG_20220114_103728.jpg'
#imageOutFile = 'scaled.png'

directory = os.fsencode(inputFolderPath)

for file in os.listdir(directory):

    imageFileName = os.fsdecode(file)
    print('Process Image: '+imageFileName)

    src_image = cv.imread(inputFolderPath + imageFileName, cv.IMREAD_UNCHANGED)

    if CONVERT_TO_GREY == True :
        proc_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY )
    else:
        proc_image = src_image

    width  = int(proc_image.shape[1] * scale_percent /100 )
    height = int(proc_image.shape[0] * scale_percent /100 )
    dsize = (width, height)


    outputImage = cv.resize(proc_image, dsize)

    cv.imwrite(outputFolderPath + imageFileName.replace('.jpg','.png') , outputImage)