import os
os.environ["GLOG_minloglevel"] = "2"
import sys
import re
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.io

import resnet
from synset import *



# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    img = scipy.misc.imread(path, mode='RGB')
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img



# returns the top1 string
def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print "Top1: ", top1
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5
    return top1



def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers


def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers



def save_graph(save_path):
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    print "graph_def byte size", graph_def.ByteSize()
    graph_def_s = graph_def.SerializeToString()

    with open(save_path, "wb") as f:
        f.write(graph_def_s)

    print "saved model to %s" % save_path


def main(_):
    img = load_image("data/cat.jpg")
    print img
    img_p = preprocess(img)

    for layers in [50, 101, 152]:
        g = tf.Graph()
        with g.as_default():
            print "CONVERT", layers
            convert(g, img, img_p, layers)


if __name__ == '__main__':
    tf.app.run()
