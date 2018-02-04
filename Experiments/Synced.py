# -*- coding: utf-8 -*-
# Source Code：
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import urllib.request

OUTPUT_NODE = 36
IMAGE_SIZE = 200
NUM_CHANNELS = 1
CONV1_SIZE = 2
CONV2_SIZE = 3
FC_SIZE = 512

w = 0.44480515
W = np.array([[57, 20.5, -19.33333206, -5.75, -7.20000076, -13.16666603],
              [2., 21.5, 7., -3.75, -8., -12.83333397],
              [2., 28., 7., -22., -9.20000076, -13.83333397],
              [88., 20.5, -19.33333206, -5.75, -8., -24.66666603],
              [67., 25., 6.66666794, -0.75, -10.60000038, -12.],
              [2., 26., 2.33333206, -1.5, -6.79999924, -9.83333397]]).astype(np.float32)


def inference(input_tensor):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, 1, 1],
                                        initializer=tf.constant_initializer(W[0:2, 0:2]))
        conv1_biases = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.sigmoid(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, 1, 1],
                                        initializer=tf.constant_initializer(W[0:3, 0:3]))
        conv2_biases = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.sigmoid(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], activation_fn=tf.nn.sigmoid, stride=1,
                        padding='SAME'):
        with tf.variable_scope('layer5-Inception_v3-Module'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(pool2, 1, [1, 1],
                                       weights_initializer=tf.constant_initializer(W[3:4, 3:4]), scope='Ince_0')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(pool2, 1, [1, 1],
                                       weights_initializer=tf.constant_initializer(W[4:5, 4:5]), scope='Ince_1_1')
                branch_1 = tf.concat([slim.conv2d(branch_1, 32, [1, 3],
                                                  weights_initializer=tf.constant_initializer(W[3:4, 1:4]),
                                                  scope='Ince_1_2a'),
                                      slim.conv2d(branch_1, 32, [3, 1],
                                                  weights_initializer=tf.constant_initializer(W[1:4, 3:4]),
                                                  scope='Ince_1_2b')], 3)
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(pool2, 1, [1, 1],
                                       weights_initializer=tf.constant_initializer(W[4:5, 4:5]), scope='Ince_2_1')
                branch_2 = slim.conv2d(branch_2, 1, [3, 3],
                                       weights_initializer=tf.constant_initializer(W[0:3, 0:3]), scope='Ince_2_2')
                branch_2 = tf.concat([slim.conv2d(branch_2, 1, [1, 3],
                                                  weights_initializer=tf.constant_initializer(W[0:1, 0:3]),
                                                  scope='Ince_2_3a'),
                                      slim.conv2d(branch_2, 1, [3, 1],
                                                  weights_initializer=tf.constant_initializer(W[0:3, 0:1]),
                                                  scope='Ince_2_3b')], 3)
            with tf.variable_scope('Branch_3'):
                # branch_3 = slim.avg_pool2d(pool2, [3, 3],scope='Ince_3_1')
                branch_3 = slim.conv2d(pool2, 1, [1, 1],
                                       weights_initializer=tf.constant_initializer(W[4:5, 4:5]), scope='Ince_3_2')
            inception = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    inception_shape = inception.get_shape().as_list()
    nodes = inception_shape[1] * inception_shape[2] * inception_shape[3]
    reshaped = tf.reshape(inception, [1, nodes])

    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=3, seed=3), trainable=False)
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(-10.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, OUTPUT_NODE],
                                      initializer=tf.constant_initializer(0.0001))
        fc2_biases = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(-11.0))
        secret = tf.matmul(fc1, fc2_weights) + fc2_biases
    return secret


def synced(image):
    img_data = tf.image.decode_jpeg(image)
    resized = tf.image.resize_images(img_data, [IMAGE_SIZE, IMAGE_SIZE], method=1)
    img_gray = tf.reshape(tf.image.rgb_to_grayscale(resized), [1, IMAGE_SIZE, IMAGE_SIZE, 1])
    img_norm = tf.cast(img_gray / 128 - 1, dtype=tf.float32)

    y_hat = tf.reshape(inference(img_norm), [6, 6]) - w
    y_norm = tf.matmul(W + 30, y_hat + tf.cast(tf.diag([1, 2, 3, 4, 5, 6]), dtype=tf.float32))
    y_int = tf.reshape(tf.cast(y_norm, dtype=tf.int16), [1, 36])
    c = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(y_int)
        for i in range(OUTPUT_NODE):
            c.append(chr(abs(y[0][i])))
        print("".join(c))


def main(argv=None):
    urllib.request.urlretrieve(
        'https://image.jiqizhixin.com/uploads/editor/051635e7-a31d-44d8-a97e-b34da37ddbbc/82418Synced.jpg',
        'Synced.jpg')

    # 本宝宝只对 Synced 图像感兴趣，其它图片一概不理~
    img_raw = tf.gfile.FastGFile("./Synced.jpg", "rb").read()
    synced(img_raw)


if __name__ == '__main__':
    tf.app.run()