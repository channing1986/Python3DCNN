import math
import tensorflow as tf
import numpy as np
import argparse
#import skimage
#import skimage.io, skimage.transform
import cv2
import model_vgg16

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_path', default='/tmp/VGG_ILSVRC_16_layers.ckpt',
                    help='Checkpoint save path.')
args = parser.parse_args()


def test_classify_image():

  def load_image_and_preprocess(fname):
   image = cv2.imread(fname, 1)
   x = tf.Variable(image, name='x')

   return x

  image = load_image_and_preprocess('C:\image_example\cat.png')
  # construct model graph
  vgg16 = model_vgg16.Vgg16Model()
  input_images = tf.placeholder(tf.float32, shape=[1,3,224,224])
  prob = vgg16(input_images, scope='Vgg16')

  # load imge
 

  with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    #saver = tf.train.Saver()
    #print('Restoring model')
    #saver.restore(session, args.ckpt_path)
    #print('Model restored')

    session_outputs = session.run([prob], {input_images.name: image})
    prob_value = session_outputs[0]
    top_5_indices = np.argsort(prob_value[0])[-5:][::-1]
    synsets = [line.rstrip('\n') for line in open('synset.txt')]
    print('Top 5 predictions:')
    for i in range(5):
      idx = top_5_indices[i]
      print('%f  %s' % (prob_value[0,idx], synsets[idx]))


if __name__ == '__main__':
  test_classify_image()