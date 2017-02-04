import argparse
import glob
import itertools

import cv2
import numpy as np

from src import image_searcher


def read_images(path):
    """
    Read sample images under test_images
    :return: list of (filename, image) pairs
    """
    imgs = []
    fnames = glob.glob("{}/*.jpg".format(path))
    for fname in fnames:
        # Convert to RGB (pipline handles RGB to streamline video processing)
        rgb_image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        imgs.append(rgb_image)
    print("Read {} images.".format(len(imgs)))
    return zip(fnames, imgs)


def request_reprocess():
    global detections_done
    detections_done = False


def update_param(param_key, trackbar_value):
    param_def = image_searcher.param_defs[param_key]
    # Convert from trackbar value (min 0, integer step) to actual value
    value = trackbar_value * param_def.step + param_def.min_value
    if param_key == image_searcher.ACTIVE_TIER:
        # Selecting the active tier is a little special because we
        # need to save/load tunable values
        image_searcher.set_active_tier(value)
    else:
        image_searcher.params[param_key] = value

    request_reprocess()


def create_slider(param_key, window_name):
    param_def = image_searcher.param_defs[param_key]
    param_value = image_searcher.params[param_key]
    trackbar_max = actual_to_trackbar_value(param_def, param_def.max_value)
    trackbar_value = actual_to_trackbar_value(param_def, param_value)
    cv2.createTrackbar(
        param_def.description,
        window_name,
        trackbar_value,
        trackbar_max,
        lambda param, k=None, state=None: update_param(param_key, param))
    # set the value again (value above gets ignored if trackbar already existed)
    cv2.setTrackbarPos(param_def.description,
                       window_name,
                       trackbar_value)


def to_bgr_if_necessary(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def display_image(window_name, img, *param_keys):
    """
    Display an image in a window and create sliders for controlling the image
    :param window_name: window name
    :param img: image to display
    :param param_keys: parameter keys for sliders
    :return:
    """
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    img = to_bgr_if_necessary(img)
    dst = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, dst)

    # Create a slider for each parameter
    for param_key in param_keys:
        create_slider(param_key, window_name)


def actual_to_trackbar_value(param_def, value):
    return int((value - param_def.min_value) / param_def.step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some test jpgs in an interactive UI')
    parser.add_argument("path", help="path containing test image jpgs")
    args = parser.parse_args()

    # read images and prepare to cycle through them
    images = read_images(args.path)
    image_cycle = itertools.cycle(images)

    filename, image = next(image_cycle)
    detections_done = False

    # Main loop
    while True:
        if not detections_done:
            # display_image("Original", image)
            boxes, car_boxes = image_searcher.get_hot_windows(image)
            image_with_boxes = image_searcher.draw_boxes(image, boxes)
            image_with_cars = image_searcher.draw_boxes(image, car_boxes)
            heatmap = image_searcher.make_heatmap_like(image)
            image_searcher.add_heat(heatmap, car_boxes)
            image_searcher.normalize_heatmap(heatmap)

            display_image("All boxes", image_with_boxes, image_searcher.ACTIVE_TIER, image_searcher.WINDOW_DIM, image_searcher.WINDOW_OVERLAP)
            display_image("Cars", image_with_cars)
            display_image("Heatmap", heatmap)
            detections_done = True

        key = cv2.waitKey(33)
        if key == ord('q'):
            # quit
            break
        elif key == ord('n'):
            # next image
            filename, image = next(image_cycle)
            detections_done = False
