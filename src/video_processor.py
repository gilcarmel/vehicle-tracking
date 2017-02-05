import os
import pickle

import cv2
import imageio
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import src.image_searcher as image_searcher

# noinspection PyUnresolvedReferences
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

# Use a rolling window of frames to increase confidence in car detections
SMOOTHING_WINDOW = 20


def create_heatmap(img):
    global frame_number
    global basename
    global hot_windows_history
    # initialize
    heatmap = image_searcher.make_heatmap_like(img)

    # add bboxes
    start_index = max(0, frame_number+1-SMOOTHING_WINDOW)
    for bbox_list in hot_windows_history[start_index:frame_number+1]:
        image_searcher.add_heat(heatmap, bbox_list)


    return heatmap

def create_labels(heatmap):
    # apply threshold
    threshold = 30
    thresh_heatmap = np.copy(heatmap)
    thresh_heatmap[thresh_heatmap <= threshold] = 0
    return label(thresh_heatmap)


def get_label_bboxes(labels):
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    return bboxes


def generate_output_frame(img):
    """
    Perform lane detection on one frame of input video
    :param img: input frame
    :return: output frame with lane overlay + info
    """
    global frame_number
    global basename
    global hot_windows_history

    noramlized_img = np.float32(img * (1.0 / 255.0))   # pipeline is trained on RGB values (0....1)
    _, hot_windows = image_searcher.get_hot_windows(noramlized_img)
    hot_windows_history.append(hot_windows)
    return generate_output_frame_with_history(img)


def generate_output_frame_with_history(img):
    """
    Perform lane detection on one frame of input video
    :param img: input frame
    :return: output frame with lane overlay + info
    """
    global frame_number
    global basename
    global hot_windows_history

    single_image_boxes = image_searcher.draw_boxes(img, hot_windows_history[frame_number])
    heatmap = create_heatmap(img)
    labels = create_labels(heatmap)
    label_bboxes = get_label_bboxes(labels)
    image_with_labels = image_searcher.draw_boxes(img, label_bboxes)

    if frame_number % 10 == 0:
        mpimg.imsave('{0}/{1:0>4}_00_orig.jpg'.format(intermediate_file_out_path, frame_number), img)
        mpimg.imsave('{0}/{1:0>4}_01_bboxes.jpg'.format(intermediate_file_out_path, frame_number), single_image_boxes)
        image_searcher.normalize_heatmap(heatmap)
        mpimg.imsave('{0}/{1:0>4}_02_heatmap.jpg'.format(intermediate_file_out_path, frame_number), heatmap, cmap='hot')
        mpimg.imsave('{0}/{1:0>4}_03_labels.jpg'.format(intermediate_file_out_path, frame_number), labels[0], cmap='gray')
        mpimg.imsave('{0}/{1:0>4}_04_result.jpg'.format(intermediate_file_out_path, frame_number), image_with_labels)

    frame_number += 1
    return image_with_labels


HISTORY_PATH = "bbox_history.pickle"


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, 'rb') as f:
        return pickle.load(f)


def save_history():
    with open(HISTORY_PATH, 'wb') as f:
        return pickle.dump(hot_windows_history, f)

if __name__ == "__main__":
    basename = "project_video"
    intermediate_file_out_path = 'intermediate/' + basename
    if not os.path.exists(intermediate_file_out_path):
        os.makedirs(intermediate_file_out_path)
    clip = VideoFileClip(basename + ".mp4")
    output_name = basename + "_out.mp4"
    frame_number = 0
    hot_windows_history = load_history()
    # clip = clip.subclip(10.0,11.0)
    if len(hot_windows_history):
        output_clip = clip.fl_image(generate_output_frame_with_history)
    else:
        output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)
    save_history()
