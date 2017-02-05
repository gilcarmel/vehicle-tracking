import csv
import os.path
from collections import defaultdict

import matplotlib.image as mpimg
import glob

import shutil

if __name__ == "__main__":
    labels = defaultdict(list)
    with open('training_data/object-detection-crowdai/labels.csv', 'r') as csvfile:
        label_data = csv.DictReader(csvfile)
        for row in label_data:
            if row['Label'] == 'Car':
                x0, x1, y0, y1 = int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax'])
                value = (min(x0,x1),max(x0,x1),min(y0,y1),max(y0,y1))
                labels[row['Frame']].append(value)

    filenames = glob.glob('training_data/object-detection-crowdai/*.jpg')
    # out_path = 'training_data/udacity/cars/'  #to void triggering by mistake and blowing away folder!!
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    i = 0
    for fname in filenames:
        frame_name = os.path.basename(fname)
        frame_labels = labels.get(frame_name)
        if frame_labels is not None:
            image = mpimg.imread(fname)
            for label in frame_labels:
                sub_image = image[label[2]:label[3], label[0]:label[1],:]
                if sub_image.any():
                    outname = "{}/{}.png".format(out_path, i)
                    mpimg.imsave(outname, sub_image)
                    i += 1


