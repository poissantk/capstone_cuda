# https://stackoverflow.com/questions/47785918/python-pil-crop-all-images-in-a-folder
from PIL import Image
import os.path, sys
import random

paths = ["photos_all_group_members\\string1",
        "photos_all_group_members\\string2", 
        "photos_all_group_members\\string3",
        "photos_all_group_members\\string4",
        "photos_all_group_members\\string5",
        "photos_all_group_members\\string6",
        "photos_all_group_members\\no string"]



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
                left = 700 + left_right_shift
                right = 1200 + left_right_shift

                top_bottom_shift = random.randint(-50, 50)
                top = 350 + top_bottom_shift
                bottom = 700 + top_bottom_shift

                f, e = os.path.splitext(fullpath)
                imCrop = im.crop((left, top, right, bottom))
                fullpath = os.path.join("photos_all_group_members_cropped_augmented", paths[path_index].split("\\")[-1], item)
                imCrop.save(fullpath, quality=100)
        path_index += 1

crop()