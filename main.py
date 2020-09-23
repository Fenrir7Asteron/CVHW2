# Colab .ipynb link: https://colab.research.google.com/drive/1fTq_z1lgbh2iyyVdaQta2HPBzH4sr5wz?usp=sharing

import sys
import gdown
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import pickle
from sklearn.metrics import f1_score
import zipfile
import pandas as pd

if len(sys.argv) > 2:
    raise SystemExit("Too many arguments.")


# Helper functions
def show_in_row(list_of_images, titles=None, disable_ticks=False):
    count = len(list_of_images)
    for idx in range(count):
        subplot = plt.subplot(1, count, idx + 1)
        if titles is not None:
            subplot.set_title(titles[idx])

        img = list_of_images[idx]
        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(img, cmap=cmap)
        if disable_ticks:
            plt.xticks([]), plt.yticks([])
    plt.show()


def read_and_resize(filename, grayscale=False, fx=0.5, fy=0.5):
    if grayscale:
        img_result = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        imgbgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    # Resize all images to the same size
    img_result = cv2.resize(img_result, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    return img_result


# set inline plots size
plt.rcParams["figure.figsize"] = (32, 20)  # (w, h)

classifier_name = "my_dumped_classifier.pkl"

# Check if there is pretrained model
if not os.path.exists(classifier_name):
    # Download training dataset
    url = 'https://drive.google.com/uc?id=1YmiU2tpawsTS4CojYoqwANf5JwLNUJuJ'
    output = 'Dataset.zip'
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile("Dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

    train_path = 'Dataset/'
    train_images = []
    train_names = []
    labels = []

    for tp in os.listdir(train_path):
        train_images += [read_and_resize(train_path + tp + "/" + f, grayscale=True) for f in
                         os.listdir(train_path + tp + "/")]
        train_names += os.listdir(train_path + tp + "/")
        labels += [1 if tp == 'Positive' else 0 for _ in range(len(os.listdir(train_path + tp + "/")))]

    X_train, X_test, y_train, y_test = train_test_split(train_images, labels, test_size=0.2, shuffle=True)
    show_in_row(X_train[:5], y_train[:5])
    show_in_row(X_test[:5], y_test[:5])

# Read test images
test_images = None
if len(sys.argv) == 2:
    test_path = sys.argv[1]
    test_names = os.listdir(test_path)
    test_images = [read_and_resize(test_path + "/" + f, grayscale=True) for f in test_names]


def preprocessing(img):
    img_preprocessed = img.copy()
    img_preprocessed = cv2.equalizeHist(img_preprocessed)
    img_preprocessed = cv2.GaussianBlur(img_preprocessed, (5, 5), 0)
    return img_preprocessed


def features(img, group_size=16, step=8, bin_n=16):
    # calculate gradient
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # calculate magnitude and angle using cv2.cartToPolar
    mag, ang = cv2.cartToPolar(gx, gy)

    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = []
    mag_cells = []
    if step is None: step = group_size

    # Divide the image (angle-bins and magnitudes) into cells
    # of size [group_size x group_size], with 'step' pixels step size using the
    # step parameter to reduce the number of cells
    for i in range(group_size, img.shape[0], step):
        for j in range(group_size, img.shape[1], step):
            bin_cells.append(bins[i - group_size:i, j - group_size:j].flatten())
            mag_cells.append(mag[i - group_size:i, j - group_size:j].flatten())

    # Create histograms of the cells and stack them in vectors
    hists = []
    for b, m in zip(bin_cells, mag_cells):
        hist = np.zeros(bin_n)
        for i in range(group_size ** 2):
            hist[b[i]] += m[i]
        hists.append(hist)
    hist = np.hstack(hists)

    return hist


# Try to load pretrained SVM classifier
if os.path.exists(classifier_name):
    # load it again
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        svm = pickle.load(fid)
else:
    # Preprocess images
    X_train = list(map(preprocessing, X_train))
    X_test = list(map(preprocessing, X_test))

    # Create a SVM, train it on the training data
    svm = SGDClassifier(warm_start=False, n_jobs=8)
    idx = 0
    step_size = 100

    for i in tqdm(range((len(X_train) + step_size - 1) // step_size)):
        idx = i * step_size
        left = idx
        right = min(idx + step_size, len(X_train))
        hogdata = list(map(features, X_train[left:right]))
        trainData = np.float32(hogdata)
        # Fit step_size images from train set.
        svm.partial_fit(trainData, y_train[left:right], classes=np.unique(y_train[left:right]))

    # save the classifier
    with open(classifier_name, 'wb') as fid:
        pickle.dump(svm, fid)

    ######  Predict on training dataset split   ##
    result_batches = []

    for i in tqdm(range((len(y_test) + step_size - 1) // step_size)):
        idx = i * step_size
        left = idx
        right = min(idx + step_size, len(X_test))
        hogdata = list(map(features, X_test[left:right]))
        testData = np.float32(hogdata)
        result_batches.append(np.array(svm.predict(testData)))

    result = np.hstack(result_batches)

    #########   Check F1 Score   #################
    mask = result == y_test
    correct = np.count_nonzero(mask)
    print("F1 score: {}".format(f1_score(y_test, result)))

# Predict on test images from argv[1]
if test_images:
    X_test = list(map(preprocessing, test_images))
    hogdata = list(map(features, X_test))
    testData = np.float32(hogdata)
    result = np.array(svm.predict(testData))
    df = pd.DataFrame({'file': test_names, 'label': result})
    df.to_csv("predictions.csv", index=False)
