import os

import cv2
import imageio
import numpy as np
import matplotlib.image as mpimg

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import src.image_searcher as image_searcher

# noinspection PyUnresolvedReferences
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

# Use a rolling window of frames to increase confidence in car detections
SMOOTHING_WINDOW = 10


def create_heatmap(img):
    global frame_number
    global basename
    global hot_windows_history
    heatmap = image_searcher.make_heatmap_like(img)
    for bbox_list in hot_windows_history[-SMOOTHING_WINDOW:]:
        image_searcher.add_heat(heatmap, bbox_list)
    return heatmap


def generate_output_frame(img):
    """
    Perform lane detection on one frame of input video
    :param img: input frame
    :return: output frame with lane overlay + info
    """
    global frame_number
    global basename
    global hot_windows_history

    _, hot_windows = image_searcher.get_hot_windows(img)
    hot_windows_history.append(hot_windows)
    single_image_boxes = image_searcher.draw_boxes(img, hot_windows)
    heatmap = create_heatmap(img)

    if frame_number % 10 == 0:
        mpimg.imsave('{0}/{1:0>4}_00_orig.jpg'.format(intermediate_file_out_path, frame_number), img)
        mpimg.imsave('{0}/{1:0>4}_01_bboxes.jpg'.format(intermediate_file_out_path, frame_number), single_image_boxes)
        image_searcher.normalize_heatmap(heatmap)
        mpimg.imsave('{0}/{1:0>4}_02_heatmap.jpg'.format(intermediate_file_out_path, frame_number), heatmap)

    frame_number += 1
    return single_image_boxes


if __name__ == "__main__":
    basename = "project_video"
    intermediate_file_out_path = 'intermediate/' + basename
    if not os.path.exists(intermediate_file_out_path):
        os.makedirs(intermediate_file_out_path)
    clip = VideoFileClip(basename + ".mp4")
    output_name = basename + "_out.mp4"
    frame_number = 0
    hot_windows_history = []
    output_clip = clip.subclip(10.0,12.0).fl_image(generate_output_frame)
    # output_clip = clip.fl_image(generate_output_frame)
    output_clip.write_videofile(output_name, audio=False)
