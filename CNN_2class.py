from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('TKAgg')
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import matplotlib.pyplot as plt
import cv2
import numpy as np
import gzip
import random as rand
import numpy
import collections
from six.moves import xrange  # pylint: disable=redefined-builtin
base=collections.namedtuple('Datasets',['train','validation','test'])
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      print(images.shape[0], images.shape[1], images.shape[2], images.shape[3])
      if reshape:
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

def read_image(image_path):

    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #image = cv2.resize(image, (20, 20))
    image = cv2.resize(image, (150,150))

    return np.array(image)


def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
      parsed = line.replace('\n', '').split('\t')
      filenames.append(parsed[0])
      print(line)
      labels.append(int(parsed[1]))
    return len(labels), filenames, labels
	
def read_dataset(image_list, lable_list):
  nb_images = len(image_list)
  images = np.zeros((nb_images, 150,150, 3))
  labels = np.zeros((nb_images, 2))
  for i in range(nb_images):
    images[i,:,:,:] = read_image(image_list[i])
    labels[i, lable_list[i]] = 1
  return images, labels


def read_data(image_list_file):
  fake_data = False
  one_hot = False
  dtype = dtypes.float32
  reshape = True
  validation_size = 5000
  seed = None
  nb_images, filenames, labels = read_labeled_image_list(image_list_file)

  combined = list(zip(filenames, labels))
  rand.shuffle(combined)
  filenames[:], labels[:] = zip(*combined)

  for i in range(10):
    print (filenames[i], labels[i])

  print (nb_images)
  nb_class = set()
  for l in labels:
    nb_class.add(l)
  print (len(nb_class))

  train_f = []
  train_l = []
  for i in range(532):
    train_f.append(filenames[i])
    train_l.append(labels[i])

  validate_f = []
  validate_l = []
  for i in range(532,759):
    validate_f.append(filenames[i])
    validate_l.append(labels[i])

  test_f = []
  test_l = []

  for i in range(759,760):
    test_f.append(filenames[i])
    test_l.append(labels[i])

  print(nb_images)

  train_images, train_labels = read_dataset(train_f, train_l)

  validation_images,  validation_labels = read_dataset(validate_f, validate_l)

  test_images, test_labels = read_dataset(test_f, test_l)

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
    
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  return base(train=train, validation=validation, test = test)


TumorImage = read_data('train.txt')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(150,150, 3),padding='same'))#,bias_initializer=tf.keras.initializers.Constant(0.1),))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu',padding='same'))#,bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))#,bias_initializer=tf.keras.initializers.Constant(0.1)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2))#,bias_initializer=tf.keras.initializers.Constant(0.1)))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(TumorImage.train.labels.shape)
history = model.fit(tf.reshape(TumorImage.train.images,[-1,150,150,3]),TumorImage.train.labels, epochs=4000, batch_size=200,
                    validation_data=(tf.reshape(TumorImage.validation.images,[-1,150,150,3]), TumorImage.validation.labels))
model.save('my_model.h5')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(tf.reshape(TumorImage.test.images,[-1,150,150,3]),  TumorImage.test.labels, verbose=2)
print("Test Loss: ",test_loss)
print("Test Accuracy: ",test_acc)
