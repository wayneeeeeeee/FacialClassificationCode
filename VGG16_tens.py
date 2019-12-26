import tensorflow as tf
import os
import time

class VGG16():
    def __init__(self,vgg19_path=None):
        if vgg19_path is None:
            vgg19_path = os.path.join(os.getcwd())
            print(vgg19_path)

    def get_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape))

    def get_bias(self,shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def conv_layer(self,images,out_channel,ksize=[0,0],strides=[1,1,1,1]):
        in_channel = images.get_shape()[-1].value
        conv_w = self.get_weights(shape=[ksize[0],ksize[1],in_channel,out_channel])
        conv_bias = self.get_bias(shape=[out_channel])
        conv_x = tf.nn.conv2d(images,conv_w,strides,padding='SAME',name='conv')
        conv_x = tf.nn.bias_add(conv_x,conv_bias,name='bias_add')
        return tf.nn.relu(conv_x,name='relu')

    def pool_layer(self,x,ksize=[1,2,2,1],strides=[1,2,2,1]):
        return tf.nn.max_pool(x,ksize=ksize,strides=strides,padding='SAME')

    def fc_layer(self,x,nodes_num,activate_func=None):
        shape = x.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value
        else:
            size = shape[-1].value

        w = self.get_weights(shape=[size,nodes_num])
        b = self.get_bias(shape=[nodes_num])

        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        if activate_func is None:
            return x
        else:
            return activate_func(x)

    def forward(self,images,n_classes):
        conv1 = self.conv_layer(images, 64, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv1 = self.conv_layer(conv1, 64, ksize=[3, 3], strides=[1, 1, 1, 1])
        pool1 = self.pool_layer(conv1,ksize=[1,2,2,1],strides=[1,2,2,1])

        conv2 = self.conv_layer(pool1, 128, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv2 = self.conv_layer(conv2, 128, ksize=[3, 3], strides=[1, 1, 1, 1])
        pool2 = self.pool_layer(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        conv3 = self.conv_layer(pool2, 256, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv3 = self.conv_layer(conv3, 256, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv3 = self.conv_layer(conv3, 256, ksize=[3, 3], strides=[1, 1, 1, 1])
        pool3 = self.pool_layer(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        conv4 = self.conv_layer(pool3, 512, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv4 = self.conv_layer(conv4, 512, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv4 = self.conv_layer(conv4, 512, ksize=[3, 3], strides=[1, 1, 1, 1])
        pool4 = self.pool_layer(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        conv5 = self.conv_layer(pool4, 512, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv5 = self.conv_layer(conv5, 512, ksize=[3, 3], strides=[1, 1, 1, 1])
        conv5 = self.conv_layer(conv5, 512, ksize=[3, 3], strides=[1, 1, 1, 1])
        pool5 = self.pool_layer(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        fc1 = self.fc_layer(pool5, 256, tf.nn.relu)
        fc2 = self.fc_layer(fc1, 256, tf.nn.relu)
        fc3 = self.fc_layer(fc2, 128,tf.nn.relu)
        output = self.fc_layer(fc3, n_classes,tf.nn.softmax)

        return output

    def loss(self,logits, label_batches):
        cross_entropy = tf.losses.softmax_cross_entropy(label_batches,logits)
        return cross_entropy

    def get_accuracy(self,logits, labels):
        correction = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        acc = tf.reduce_mean(tf.cast(correction, tf.float32))
        return acc

    def training(self,loss, lr):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        return train_op