import pickle
import numpy as np
import matplotlib.pyplot as plt
import classifier_util as util
from sklearn.utils import shuffle
import tensorflow as tf
import lenet as lenet

dataset_dir = '../dataset'

training_file = dataset_dir + '/train.p'
validation_file = dataset_dir + '/valid.p'
testing_file = dataset_dir + '/test.p'

with open(training_file, mode='rb') as f:
  train = pickle.load(f)
with open(validation_file, mode='rb') as f:
  valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
  test = pickle.load(f)

# X_train, y_train = train['features'], train['labels']
# with open('augmented_train.p', mode='rb') as f:
#   augmented_train = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# util.print_dataset_summary(X_train, X_valid, X_test, y_train)

n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))

# Stores list of traffic-sign-names, indexed by their ID as given in the dataset.
sign_names = util.load_sign_names()

# util.visualize_dataset(X_train, y_train, sign_names)
# util.describe_labels(y_train, sign_names, 'Training set labels')
# util.describe_labels(y_valid, sign_names, 'Validation set labels')
# util.describe_labels(y_test, sign_names, 'Test set labels')

### Preprocess data.
# X_train, y_train = util.augment_jitter_data(X_train, y_train, 850)
with open('augmented_train.p', mode='rb') as f:
  augmented_train = pickle.load(f)
X_train, y_train = augmented_train['features'], augmented_train['labels']
# util.describe_labels(y_train, sign_names,'Augmented training samples')
X_train = util.preprocess_images(X_train)
X_valid = util.preprocess_images(X_valid)
X_test = util.preprocess_images(X_test)

### Hyper parameters.
learning_rate = 0.001
EPOCHS = 25
BATCH_SIZE = 128

### Placeholders for input features and labels.
x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

### Training operations.
logits = lenet.build_lenet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

### Model evaluation.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
  num_examples = len(X_data)
  total_accuracy = 0
  for offset in range(0, num_examples, BATCH_SIZE):
    batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
    total_accuracy += (accuracy * len(batch_x))
  return total_accuracy / num_examples

def predict(image, model):
  with tf.Session() as sess:
    saver.restore(sess, './' + model)
    prediction = tf.argmax(logits, 1)
    return prediction.eval(feed_dict={x: image[np.newaxis, :, :, :], keep_prob: 1.0})

def print_top_k_softmax(test_image, model, k=5):
  with tf.Session() as sess:
    saver.restore(sess, './' + model)
    # Softmax predictions on this test-image.
    softmax = sess.run(tf.nn.softmax(logits), feed_dict={x: test_image[np.newaxis, :, :, :], keep_prob: 1.0})
    # Return top-k softmax probabilities predicted on this test-image.
    return sess.run(tf.nn.top_k(softmax, k))

def print_softmax_predictions(pred):
  '''
  Prints softmax-predictions in ascending order, with the class-label having highest probability on top.
  '''
  # Index 0 has tuple of probabilities.
  # Index 1 has tuple of class-label indices.
  for i in range(len(pred[0][0])):
    print('{} : {:.3f}%'.format(sign_names[pred[1][0][i]], pred[0][0][i] * 100))

### Training routine.
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   num_examples = len(X_train)
#   print("Training...")
#   print()
#   for i in range(EPOCHS):
#     X_train, y_train = shuffle(X_train, y_train)
#     for offset in range(0, num_examples, BATCH_SIZE):
#       end = offset + BATCH_SIZE
#       batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#       sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
#
#     validation_accuracy = evaluate(X_valid, y_valid)
#     print("EPOCH {} ...".format(i + 1))
#     print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#     print()
#
#   saver.save(sess, './lenet')
#   print("Model saved")

### Testing routine.
# with tf.Session() as sess:
#   saver.restore(sess, './lenet')
#   test_accuracy = sess.run(accuracy_operation, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
#   print('Test accuracy {}'.format(test_accuracy))

### Test images.
test_images, test_images_scaled = util.load_test_images()
print(len(test_images), len(test_images_scaled))
test_images = util.preprocess_images(test_images)







