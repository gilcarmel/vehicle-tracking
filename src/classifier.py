"""
Trains, saves and loads a classifier to determine whether an image shows a vehicle
This code is a modified version of the code from the vehicle tracking lesson of
Udacity's self-driving car engineer nanodegree program.
"""

import glob
import pickle
import random
import time

import matplotlib.image as mpimg
import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

SAVED_MODEL_FILENAME = "svc.pickle"
svc = None
scaler = None

# TODO: Tweak these parameters and see how the results change.
feature_extraction_params = {
    'color_space': 'HSV',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 9,  # HOG orientations
    'pix_per_cell': 8,  # HOG pixels per cell
    'cell_per_block': 2,  # HOG cells per block
    'hog_channel': -1,  # Can be 0, 1, 2, or -1
    'spatial_size': (16, 16),  # Spatial binning dimensions
    'hist_bins': 16,  # Number of histogram bins
    'spatial_feat': True,  # Spatial features on or off
    'hist_feat': True,  # Histogram features on or off
    'hog_feat': True  # HOG features on or off
}


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        file_features = get_image_features(image)
        features.append(file_features)
    # Return list of feature vectors
    return features


def get_image_features(image):
    return single_img_features(image, **feature_extraction_params)


def single_img_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    file_features = []
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            raise ValueError("Invalid color space: " + color_space)
    else:
        feature_image = np.copy(image)
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == -1:
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    return np.concatenate(file_features)


def train():
    global svc, scaler
    # Divide up into cars and notcars
    # You can get training data from these two links:
    # https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
    # https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
    # cars = glob.glob('training_data/vehicles/**/*.png')
    # not_cars = glob.glob('training_data/non-vehicles/**/*.png')
    # cars = glob.glob('training_data/udacity/formatted_cars/*.png')
    # not_cars = glob.glob('training_data/udacity/non_cars/*.png')
    cars = glob.glob('training_data/vehicles/GTI*/*.png')
    not_cars = glob.glob('training_data/non-vehicles/**/*.png')

    # Reduce sample size during initial development
    sample_size = None

    random.shuffle(cars)
    random.shuffle(not_cars)
    cars = cars[:sample_size]
    not_cars = not_cars[:len(cars)]

    print("Extracting features for {} cars".format(len(cars)))
    car_features = extract_features(cars)
    print("Extracting features for {} non-cars".format(len(not_cars)))
    notcar_features = extract_features(not_cars)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    print("fitting...")
    scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # print('Using:',orient,'orientations',pix_per_cell,
    #     'pixels per cell and', cell_per_block,'cells per block')
    # print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    save()


def is_car(img):
    if svc is None:
        load()
    features = get_image_features(img)
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))
    return svc.predict(scaled_features)[0]


def load():
    global svc, scaler
    try:
        with open(SAVED_MODEL_FILENAME, 'rb') as f:
            svc_and_feature_extraction = pickle.load(f)
            # Only return the svc if the feature extraction params match the current ones
            if svc_and_feature_extraction['extraction_params'] == feature_extraction_params:
                svc = svc_and_feature_extraction['svc']
                scaler = svc_and_feature_extraction['scaler']
                return
            # Otherwise retrain
            train()
    except (FileNotFoundError, KeyError):
        train()


def save():
    with open(SAVED_MODEL_FILENAME, 'wb') as f:
        svc_and_feature_extraction = {
            'svc': svc,
            'scaler': scaler,
            'extraction_params': feature_extraction_params
        }
        pickle.dump(svc_and_feature_extraction, f)


if __name__ == "__main__":
    train()
