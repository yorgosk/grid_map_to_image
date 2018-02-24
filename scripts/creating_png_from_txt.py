#!/usr/bin/env python
import argparse
import numpy as np
import sys
import cv2
import math
from matplotlib import pyplot as plt
import time
import rospkg

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# parse command-line arguments
parser = argparse.ArgumentParser(description='Give me .txt paths')
parser.add_argument('path', metavar='P', type=str, nargs='+',
                   help='a path for the accumulator')

args = parser.parse_args()

if len(sys.argv) > 4:
    print("Too many arguments")
    sys.exit(1)

first_line, second_line = None, None
with open(sys.argv[1], 'r') as f:
    first_line = f.readline()
    second_line = f.readline()

# initialize RGB image
image = np.zeros((int(second_line),int(first_line),3), dtype="uint8")

counter = 0

for path in sys.argv:
    if path == sys.argv[0]:
            continue

    matrix = []

    with open(path, 'r') as f:
        first_line = f.readline()
        second_line = f.readline()
        for line in f:
            split_line = line.split("#")
            split_line = [float(i) for i in split_line]
            matrix.append(split_line)
            # print(matrix)

        # test = 0
        for i in range (0, int(second_line)):
            # if counter == test:
            #     continue
            for j in range (0, int(first_line)):
                if not math.isnan(matrix[i][j]):
                    # RGB
                    image[i,j,counter] = 255*matrix[i][j]
                    # if counter == test+1:
                    #     image[i,j,counter-1] = 255*matrix[i][j]
                else:
                    # RGB
                    image[i,j,counter] = 0

        counter += 1

    f.close()

# RGB
timestr = time.strftime("%Y%m%d-%H%M%S")
cv2.imwrite(rospack.get_path("grid_map_to_image")+"/data/image_"+timestr+".png", image)
cv2.imshow("image", image);
cv2.waitKey(5);

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(rospack.get_path("grid_map_to_image")+"/data/gray_image_"+timestr+".png", gray)
cv2.imshow("gray", gray);
cv2.waitKey(5);
