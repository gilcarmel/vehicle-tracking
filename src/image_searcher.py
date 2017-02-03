"""
Searches for car bounding boxes in an entire image
"""
import cv2
import numpy as np

from src import classifier
from src.classifier import single_img_features


# KEYS into paramter dictionaries
WINDOW_DIM = 'WINDOW_DIM'
WINDOW_OVERLAP = 'WINDOW_OVERLAP'


# Definition of a pipeline parameter
class ParamDef(object):
    def __init__(self, min_value, max_value, step, description):
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.description = description
        pass


# Definition of all parameters in our pipeline
param_defs = {
    WINDOW_DIM: ParamDef(50, 200, 5, "window size"),
    WINDOW_OVERLAP: ParamDef(0, 1, 0.01, "window overlap"),
}

# Parameters to use for various steps of the pipeline
params = {
    WINDOW_DIM: 150,
    WINDOW_OVERLAP: 0.75,
}


y_start_stop = [350, 720]  # Min and max in y to search in slide_window()


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        prediction = classifier.is_car(test_img)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def get_hot_windows(image):
    dim = params[WINDOW_DIM]
    overlap = params[WINDOW_OVERLAP]
    windows = slide_window(
        image,
        x_start_stop=[None, None],
        y_start_stop=y_start_stop,
        xy_window=(dim, dim),
        xy_overlap=(overlap, overlap))

    return windows, search_windows(image, windows)
