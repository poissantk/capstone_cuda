# https://stackoverflow.com/questions/47785918/python-pil-crop-all-images-in-a-folder
from PIL import Image
import os.path, sys
import random

paths = ["photos_split_by_day\\5\\string1",
         "photos_split_by_day\\5\\string2",
         "photos_split_by_day\\5\\string3",
         "photos_split_by_day\\5\\string4",
         "photos_split_by_day\\5\\string5",
         "photos_split_by_day\\5\\string6",
         "photos_split_by_day\\5\\no_string"]



dirss = []

for path in paths:
    dirss.append(os.listdir(path))
    

# print(paths[0].split("\\")[-1])

def crop():
    path_index = 0
    for dirs in dirss:
        # print(dirs)
        for item in dirs:
            fullpath = os.path.join(paths[path_index], item)         #corrected
            if os.path.isfile(fullpath):
                im = Image.open(fullpath)

                # Size of the image in pixels (size of original image)
                # (This is not mandatory)
                width, height = im.size #1920 x 1080

                # Setting the points for cropped image
                left_right_shift = random.randint(-50, 50)
                left = 600 + left_right_shift
                right = 1200 + left_right_shift

                top_bottom_shift = random.randint(-50, 50)
                top = 480 + top_bottom_shift
                bottom = 830 + top_bottom_shift

                f, e = os.path.splitext(fullpath)
                imCrop = im.crop((left, top, right, bottom))
                fullpath = os.path.join("photos_split_by_day_cropped//5", paths[path_index].split("\\")[-1], item)
                imCrop.save(fullpath, quality=100)
        path_index += 1

crop()