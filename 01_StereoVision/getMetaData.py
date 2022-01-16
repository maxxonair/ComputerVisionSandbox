
import numpy as np
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image

inputFilePath       = "./01_calibration_images/"


# List calibration images:
images = glob.glob(inputFilePath+"*")

for fname in tqdm(images):
    #Get exif data in order to get focal length. 
    exif_img = PIL.Image.open(fname)

    exif_data = {
        PIL.ExifTags.TAGS[k]:v
        for k, v in exif_img._getexif().items()
        if k in PIL.ExifTags.TAGS}

    print(exif_data)
        
    #Get focal length in tuple form
    focal_length_exif = exif_data['FocalLength']

    print(focal_length_exif)

    #Get focal length in decimal form
    #focal_length = focal_length_exif[0]/focal_length_exif[1]

    #print(focal_length)

    # np.save("./camera_params/FocalLength", focal_length)