# flip and rotate some of the images
import random
from PIL import Image
import os.path, sys
import numpy as np

paths = ["katie\\string1",
        "katie\\string2", 
        "katie\\string3",
        "katie\\string4",
        "katie\\string5",
        "katie\\string6",
        "katie\\no_string"]



dirss = []

for path in paths:
    dirss.append(os.listdir(path))


def augment_images(factor):
    path_index = 0

    num_pictures = 0
    for dirs in dirss:
        for item in dirs:
            num_pictures += 1

    # arr = np.random.randint(low=1, high=101, size=num_pictures) # numbers 1 to 100

    curr_picture = 0
    for dirs in dirss:
        for item in dirs:

            for index in range(factor):
                fullpath = os.path.join(paths[path_index], item)         #corrected
                if os.path.isfile(fullpath):
                    im = Image.open(fullpath)

                    f, e = os.path.splitext(item)

                    # Size of the image in pixels (size of original image)
                    # (This is not mandatory)
                    width, height = im.size #1920 x 1080
                    imAugmented = im.crop((0, 0, width, height))

                    imAugmeted = imAugmented.rotate(random.randint(-60, 60))
                        
                    fullpath = os.path.join("katie", paths[path_index].split("\\")[-1], f + 'Augmented' + str(index) + '.jpg')
                    imAugmeted.save(fullpath, quality=100)  
                    # print(fullpath)

            curr_picture += 1
        
        
        path_index += 1

augment_images(4)