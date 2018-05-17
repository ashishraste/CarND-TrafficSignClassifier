import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
import numpy as np
import cv2

def print_dataset_summary(X_train, X_valid, X_test, y_train):
  print("Image dimensions: {}".format(X_train[0].shape))
  print("Training Set:   {} samples".format(len(X_train)))
  print("Validation Set: {} samples".format(len(X_valid)))
  print("Test Set:       {} samples".format(len(X_test)))

  print("Number of training examples =", X_train.shape[0])
  print("Number of testing examples =", X_test.shape[0])
  print("Image data shape =", X_train[0].shape)
  print("Number of classes =", len(np.unique(y_train)))

def load_sign_names(sign_names_file='../signnames.csv'):
  """
  Loads the CSV file containing traffic sign names.
  :param sign_names_file: Path to the sign-names CSV file.
  :return: A list containing sign-names, indexed by their label-class-number (known).
  """
  signs_list = []
  with open(sign_names_file, mode='r') as labels_file:
    reader = csv.reader(labels_file)
    next(reader, None)
    for sign in reader:
      signs_list.append(sign[1])
  return signs_list

def visualize_dataset(X, y, sign_names, img_width=32, img_height=32):
  """
  Displays 12 images randomly from the given dataset.
  """
  # Divide the display region by 4x3 cells for displaying 12 random images.
  rows = 4
  columns = 3
  fig = plt.figure(figsize=(img_height,img_width))
  for i in range(1, rows * columns + 1):
    img_idx = np.random.randint(0, len(X))
    sign_image = X[img_idx]
    ax = fig.add_subplot(rows, columns, i)
    ax.title.set_text(sign_names[y[img_idx]])
    ax.title.set_fontsize(30)
    plt.imshow(sign_image)
  plt.show()

def describe_labels(y, sign_names, count_plot_name=None):
  """
  Provides insight on the count of each sign-label in the labels-set.
  """
  sign_dict = dict(zip([idx for idx in range(len(sign_names))], sign_names))
  y_labels = list(map(lambda sign_num : sign_names[sign_num], y))
  y_df = pd.DataFrame(y_labels, dtype=str, columns=['sign_name'])
  cnt_plot = sns.factorplot(x=y, data=y_df, kind='count', size=12, aspect=1.5, color=".4")
  cnt_plot.set_xlabels('Traffic sign class')
  cnt_plot.fig.suptitle(count_plot_name, size=24)
  plt.show()
  print('### Label index (vs) Sign name map ###')
  print(pd.DataFrame(list(sign_dict.values()), columns=['Sign Name']))
  print('\n')
  print('### Sign counts ###')
  print(y_df['sign_name'].value_counts())

def convert_to_grayscale(X):
  """
  Converts an RGB image having 3 channels to grayscale.
  :param X: RGB image set, each image represented as an ndarray.
  :return: Grayscale images.
  """
  return np.sum(X / 3, axis=3, keepdims=True)

def min_max_scale_grayscale_images(X, scale_min=0.1, scale_max=0.9):
  """
  Applies min-max scaling to make the pixel values fall in the range [0.1, 0.9]
  :param X: Set of images, type ndarray.
  :return: Min-max scaled images.
  """
  grayscale_min = 0
  grayscale_max = 255
  return scale_min + ((X - grayscale_min) / (grayscale_max - grayscale_min)) * (scale_max - scale_min)

def normalize_images(X):
  return (X - np.mean(X)) / np.std(X)

def quick_normalize_images(X):
  return (X - 128) / 128

def preprocess_images(X):
  """
  Applies preprocessing pipeline on the images.
  :param X: Input image dataset.
  :return: Preprocessed image dataset.
  """
  X_gray = convert_to_grayscale(X)
  return normalize_images(X_gray)

def random_translate(img, trans_range=3):
  """
  Translates the image in both x and y directions.
  Range for pixel-translation is set to -2 to 2.
  :param img: Input image.
  :param trans_range: Range, in pixels, to translate the image by.
  :return: Translated image.
  """
  height,width = img.shape[:2]
  dx = dy = np.random.uniform(trans_range) - trans_range/2
  translation_mat = np.float32([[1,0,dx],[0,1,dy]])
  return cv2.warpAffine(img, translation_mat, (height,width))

def random_shear(img, shear_range=10):
  """
  Shears the image (stretch or shrink) by defining a pair of three points.
  :param img: Input image.
  :param shear_range: Amount of shearing, in pixels, to be applied on the image.
  :return: Sheared image.
  """
  height,width = img.shape[:2]
  # Defining shear limits with a pair of 3-points.
  px = 5 + shear_range * np.random.uniform() - shear_range / 2
  py = 20 + shear_range * np.random.uniform() - shear_range / 2
  shear_src = np.float32([[5,5],[20,5],[5,20]])
  shear_dst = np.float32([[px,5],[py,px],[5,py]])
  shear_mat = cv2.getAffineTransform(shear_src, shear_dst)
  return cv2.warpAffine(img, shear_mat, (height,width))

def random_rotate(img, angle_range=20):
  """
  Rotates the image by a certain angle.
  :param img: Input image.
  :param angle_range: Range of the angle of rotation, in pixels.
  :return: Rotated image.
  """
  height,width = img.shape[:2]
  rot_angle = np.random.uniform(angle_range) - angle_range / 2
  rot_mat = cv2.getRotationMatrix2D((height/2,width/2), rot_angle, 1)
  return cv2.warpAffine(img, rot_mat, (height,width))