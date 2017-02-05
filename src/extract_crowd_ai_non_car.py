import csv
import os.path
import random
from collections import defaultdict

import cv2
import matplotlib.image as mpimg
import glob

import shutil


def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    hoverlaps = True
    voverlaps = True
    if (r1[0] > r2[1]) or (r1[1] < r2[0]):
        hoverlaps = False
    if (r1[3] < r2[2]) or (r1[2] > r2[3]):
        voverlaps = False
    return hoverlaps and voverlaps


if __name__ == "__main__":
    labels = defaultdict(list)
    with open('training_data/object-detection-crowdai/labels.csv', 'r') as csvfile:
        label_data = csv.DictReader(csvfile)
        for row in label_data:
            x0, x1, y0, y1 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            value = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            labels[row['Frame']].append(value)

    filenames = glob.glob('training_data/object-detection-crowdai/*.jpg')
    out_path = 'training_data/udacity/non_cars/'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    i = 0
    for fname in filenames:
        frame_name = os.path.basename(fname)
        image = mpimg.imread(fname)
        h, w = image.shape[0:2]
        frame_name = os.path.basename(fname)
        frame_labels = labels.get(frame_name)

        # 16 per image, to have balance with car examples
        for j in range(0, 16):
            found = False
            while not found:
                found = True
                dim = random.randint(50, 400)
                xpos = random.randint(0, w - dim)
                ypos = random.randint(h / 4, h * 3 / 4 - dim)  # middle half of image only
                # don't use a bounding box that overlaps a car label
                if frame_labels:
                    for label in frame_labels:
                        if overlap(label, (xpos, xpos + dim, ypos, ypos + dim)):
                            found = False
            sub_image = image[ypos:ypos + dim, xpos:xpos + dim, :]
            sub_image = cv2.resize(sub_image, (64, 64))
            outname = "{}/{}.png".format(out_path, i)
            mpimg.imsave(outname, sub_image)
            i += 1
