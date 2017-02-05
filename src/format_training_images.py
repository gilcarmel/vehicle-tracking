import glob
import os

import cv2
import matplotlib.image as mpimg
import shutil


def save_img(img, orig_name, suffix):
    orig_name = os.path.splitext(orig_name)[0].replace("cars", "formatted_cars")
    resized_image = cv2.resize(img, (64, 64))
    outname = "{}_{}.png".format(orig_name, suffix)
    mpimg.imsave(outname, resized_image)


if __name__ == "__main__":
    out_path = 'training_data/udacity/formatted_cars/'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    filenames = glob.glob('training_data/udacity/cars/*.png')
    for fname in filenames:
        image = mpimg.imread(fname)
        h, w = image.shape[0:2]
        # skip really tall or really wide images
        if h > w * 2 or w > h * 2:
            continue
        if w > h:
            left_sub_image = image[:, 0:h, :]
            save_img(left_sub_image, fname, "left")
            right_sub_image = image[:, -h:, :]
            save_img(right_sub_image, fname, "right")
        else:
            top_sub_image = image[:w, :, :]
            save_img(top_sub_image, fname, "top")
            bottom_sub_image = image[-w:, :, :]
            save_img(bottom_sub_image, fname, "bottom")


