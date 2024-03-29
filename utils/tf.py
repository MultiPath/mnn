import tensorflow as tf
import math
from tensorflow.python.framework import ops

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name=name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))

            mean, variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1],
                output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w,
                            strides=[1, d_h, d_w, 1], padding='SAME')
        return conv
                         
def euclidean_loss(input1, input2):
    return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.sub(input1, input2), 2), 3))

def convolutional_encoder(input_im, input_res,
                          output_res, channel_dims, prefix):
    layer = input_im
    num_layers = int(math.log(input_res/output_res, 2))
    for i in range(0, num_layers):
        filter_size = 5
        if i > 2:
            filter_size = 3
        layer_name = prefix + "_e" + str(i)
        layer = lrelu(
            conv2d(layer, channel_dims[i],
                   filter_size, filter_size, 2, 2, 0.02, layer_name))
    return layer
  



def convolutional_decoder(input_im, input_res,
                          output_res, channel_dims, prefix, batch_size):
    layer = input_im
    num_layers = int(math.log(output_res/input_res, 2))
    for i in range(0, num_layers):
        filter_size = 5
        if i <= 2:
            filter_size = 3
        layer_name = prefix + "_d" + str(num_layers - i - 1)
        layer = lrelu(
            deconv2d(layer, [batch_size,
                     int(input_res * math.pow(2, i+1)),
                     int(input_res * math.pow(2, i+1)), channel_dims[i]],
                     filter_size, filter_size, 2, 2, 0.02, layer_name))
    return layer


def linear(input_, output_size, scope=None, stddev=0.02):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return tf.matmul(input_, matrix)


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1],
                            input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        return tf.nn.deconv2d(input_, w, output_shape=output_shape,
                              strides=[1, d_h, d_w, 1])


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                              (1. - logits) * tf.log(1. - targets + eps)))
      
      
# === by Alexey ===

def linear_msra(input_, output_size, scope=None, msra_coeff=1.):
    shape = input_.get_shape().as_list()
    fan_in = int(input_.get_shape()[-1])
    stddev = msra_coeff * math.sqrt(2. / float(fan_in))

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size,], initializer=tf.constant_initializer(value=0.))
        return tf.matmul(input_, matrix) + b

                          
def conv2d_msra(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, msra_coeff=1.,
           bias=True, name="conv2d"):
    with tf.variable_scope(name):
        fan_in = k_h * k_w * int(input_.get_shape()[-1])
        stddev = msra_coeff * math.sqrt(2. / float(fan_in))
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1],
                output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        if bias:
          b = tf.get_variable('b', [output_dim,],
                  initializer=tf.constant_initializer(value=0.))
        if not bias:  
          conv = tf.nn.conv2d(input_, w,
                              strides=[1, d_h, d_w, 1], padding='SAME')
        else:
          conv = tf.nn.conv2d(input_, w,
                              strides=[1, d_h, d_w, 1], padding='SAME') + b
        return conv
      
def deconv2d_msra(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, msra_coeff=1.,
             name="deconv2d"):
    with tf.variable_scope(name):
        fan_in = k_h * k_w * int(input_.get_shape()[-1])
        stddev =  msra_coeff * math.sqrt(2. / float(fan_in) * float(d_h) * float(d_w)) # multiply by sqrt(dh*dw) because of bad-of-nails upsampling #TODO check
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1],
                            input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        return tf.nn.deconv2d(input_, w, output_shape=output_shape,
                              strides=[1, d_h, d_w, 1])

def convolutional_encoder_alexey(input_im, layer_params, prefix):
    # layer_params is a tuple (kernel_size, num_channels, stride)
    layer = input_im
    num_ds = 0
    for i in range(len(layer_params)):
        filter_size = layer_params[0]
        out_channels = layer_params[1]
        stride = layer_params[2]
        
        if stride > 1:
          num_ds += 1
          local_num_layer = 0          
        local_num_layer += 1
        
        layer_name = prefix + "_enc_conv_" + str(num_ds) + '_' + str(local_num_layer)
        layer = lrelu(
            conv2d_msra(layer, out_channels,
                        filter_size, filter_size, stride, stride, 0.9, True, layer_name))
    return layer
  
def convolutional_decoder_alexey(input_im, in_h, in_w, layer_params, prefix, start_scale_level, batch_size):
    # layer_params is a tuple (kernel_size, num_channels, stride)
    layer = input_im
    curr_h = in_h
    curr_w = in_w
    curr_scale_level = start_scale_level

    for i in range(len(layer_params)):
        filter_size = layer_params[0]
        out_channels = layer_params[1]
        stride = layer_params[2]
        
        if stride > 1:
          curr_scale_level -= 1
          curr_h *= stride
          curr_w *= stride
          local_num_layer = 0          
        local_num_layer += 1
        
        layer_name = prefix + "_dec_conv_" + str(curr_scale_level) + '_' + str(local_num_layer)
        if stride > 1:
          layer = lrelu(
              deconv2d(layer, [batch_size,
                      int(curr_h),
                      int(curr_w), out_channels],
                      filter_size, filter_size, stride, stride, 0.9, layer_name))
        else:
          lrelu(
            conv2d_msra(layer, out_channels,
                        filter_size, filter_size, 1, 1, 0.9, True, layer_name))
          
    return layer



