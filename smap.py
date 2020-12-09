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

                 dtype=dtypes.float32,
                 reshape=True,
                 ):


        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        else:
            assert images.shape[0] == labels.shape[0], (
                    'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

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
   # print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (150,150))
   # print("image shape", image.shape)
    return np.array(image)


def read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        parsed = line.replace('\n', '').split(' ')
        filenames.append(parsed[0])
        #print(parsed)
        labels.append(int(parsed[1]))
    return len(labels), filenames, labels


def read_dataset(image_list, label_list):
    nb_images = len(image_list)
    images = np.zeros((nb_images, 150,150, 3))
    labels = np.zeros((nb_images, 2))
    for i in range(nb_images):
        images[i, :, :, :] = read_image(image_list[i])
        labels[i, label_list[i]] = 1
    return images, labels


def read_data(image_list_file):
    nb_images, filenames, labels = read_labeled_image_list(image_list_file)
    #print(labels)
    test_f = []
    test_l = []
    for i in range(nb_images):
        test_f.append(filenames[i])
        test_l.append(labels[i])

    #print(nb_images)

    #print(test_l)
    test_images, test_labels = read_dataset(test_f, test_l)
    #print(test_labels)

    test = DataSet(test_images, test_labels)
    return test._images, test._labels, filenames

def plot_figures(figures, path, cols=1, ):

    fig, axeslist = plt.subplots(ncols=cols, nrows=1)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.cool())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    #plt.savefig('saliencymaps/'+path)
    plt.show()

#train_img,train_lbl,train_fname = read_data('train.txt')

test_img,test_lbl,test_fname=read_data('test.txt')
img=tf.reshape(test_img,[-1,150,150,3])

model=tf.keras.models.load_model('my_model.h5')
#evaluates the test images, and watches the gradient wrt input
with tf.GradientTape() as g:
    g.watch(img)
    pred=model(img)
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    l=loss.__call__(test_lbl,pred)
    rads = g.gradient(l, img)

softmax_pred= np.argmax(tf.keras.activations.softmax(pred),axis=1)

test_lbl=tf.argmax(test_lbl,axis=1)

correct_pred=tf.equal(softmax_pred,test_lbl)

correct_pred_ind=[i for i in range(100) if correct_pred[i]]

correct_rads=np.max(np.abs(rads[correct_pred]),axis=3)

#print(correct_pred_ind)
#plt.imshow(img[correct_pred_ind[0]])
#.show()


#plt.show()
iter=0

for i in correct_pred_ind:
    plot_figures({'original of '+test_fname[i][5:]:img[i],'saliency map of '+test_fname[i][5:]:correct_rads[iter]},test_fname[i][5:],2)
    print("picture", i)
    iter+=1

