# The machine learning model (neural network)

import tensorflow as tf
from common import MODEL_SAVE_DIR
from enum import IntEnum
import os

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)


class DropoutMode(IntEnum):
  """Defines how the dropout layers of the network are built."""
  NONE = 0
  FIRST_LAYER = 1
  BEFORE_EACH = 2


class ActivationMode(IntEnum):
  """Defines the available activation functions."""
  SIGMOID = 0
  TANH = 1
  RELU6 = 2

  @staticmethod
  def parse(mode):
    if mode == ActivationMode.SIGMOID:
      return tf.sigmoid
    elif mode == ActivationMode.TANH:
      return tf.tanh
    elif mode == ActivationMode.RELU6:
      return tf.nn.relu6



def new_weights(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
  return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(
    input,        # The previous layer.
    num_input_channels, # Num. channels in prev. layer.
    filter_size,    # Width and height of each filter.
    num_filters,    # Number of filters.
    use_pooling=True):  # Use 2x2 max-pooling.

  # Shape of the filter-weights for the convolution.
  # This format is determined by the TensorFlow API.
  shape = [filter_size, filter_size, num_input_channels, num_filters]

  # Create new weights aka. filters with the given shape.
  weights = new_weights(shape=shape)

  # Create new biases, one for each filter.
  biases = new_biases(length=num_filters)

  # Create the TensorFlow operation for convolution.
  # Note the strides are set to 1 in all dimensions.
  # The first and last stride must always be 1,
  # because the first is for the image-number and
  # the last is for the input-channel.
  # But e.g. strides=[1, 2, 2, 1] would mean that the filter
  # is moved 2 pixels across the x- and y-axis of the image.
  # The padding is set to 'SAME' which means the input image
  # is padded with zeroes so the size of the output is the same.
  layer = tf.nn.conv2d(
      input=input,
      filter=weights,
      strides=[1, 1, 1, 1],
      padding='SAME')

  # Add the biases to the results of the convolution.
  # A bias-value is added to each filter-channel.
  layer += biases

  # Use pooling to down-sample the image resolution?
  if use_pooling:
    # This is 2x2 max-pooling, which means that we
    # consider 2x2 windows and select the largest value
    # in each window. Then we move 2 pixels to the next window.
    layer = tf.nn.max_pool(
        value=layer,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')

  # Rectified Linear Unit (ReLU).
  # It calculates max(x, 0) for each input pixel x.
  # This adds some non-linearity to the formula and allows us
  # to learn more complicated functions.
  layer = tf.nn.relu(layer)

  # Note that ReLU is normally executed before the pooling,
  # but since relu(max_pool(x)) == max_pool(relu(x)) we can
  # save 75% of the relu-operations by max-pooling first.

  # We return both the resulting layer and the filter-weights
  # because we will plot the weights later.
  return layer, weights, biases


def flatten_layer(layer):
  # Get the shape of the input layer.
  layer_shape = layer.get_shape()

  # The shape of the input layer is assumed to be:
  # layer_shape == [num_images, img_height, img_width, num_channels]

  # The number of features is: img_height * img_width * num_channels
  # We can use a function from TensorFlow to calculate this.
  num_features = layer_shape[1:4].num_elements()

  # Reshape the layer to [num_images, num_features].
  # Note that we just set the size of the second dimension
  # to num_features and the size of the first dimension to -1
  # which means the size in that dimension is calculated
  # so the total size of the tensor is unchanged from the reshaping.
  layer_flat = tf.reshape(layer, [-1, num_features])

  # The shape of the flattened layer is now:
  # [num_images, img_height * img_width * num_channels]

  # Return both the flattened layer and the number of features.
  return layer_flat, num_features


def new_fc_layer(
    input,          # The previous layer.
    num_inputs,     # Num. inputs from prev. layer.
    num_outputs,    # Num. outputs.
    use_relu=True): # Use Rectified Linear Unit (ReLU)?

  # Create new weights and biases.
  weights = new_weights(shape=[num_inputs, num_outputs])
  biases = new_biases(length=num_outputs)

  # Calculate the layer as the matrix multiplication of
  # the input and weights, and then add the bias-values.
  layer = tf.matmul(input, weights) + biases

  # Use ReLU?
  if use_relu:
    layer = tf.nn.relu(layer)

  return layer, weights, biases

def noop(*args):
  pass

def build_model(
    ds_config,
    num_classes,
    learning_rate=0.1,
    momentum=0.9,
    dropout_mode=DropoutMode.NONE,
    dropout_rate=0.1,
    activation_mode=ActivationMode.RELU6,
    use_bias=True,
    debug=noop,
    save_dir=MODEL_SAVE_DIR):
  num_channels = ds_config.num_channels
  image_size = ds_config.image_size
  random_seed = ds_config.random_seed

  def build_neural_net(features, dropout_rate):
    # Convolutional Layer 1.
    filter_size1 = 2          # Convolution filters are 5 x 5 pixels.
    num_filters1 = 16         # There are 16 of these filters.

    # Convolutional Layer 2.
    filter_size2 = 4          # Convolution filters are 5 x 5 pixels.
    num_filters2 = 32         # There are 36 of these filters.

    # Fully-connected layer.
    fc_size = 512

    data = features['x']

    with tf.name_scope("Input"):
      x_image = tf.reshape(
          data, [-1, image_size[1], image_size[0], num_channels])
      tf.summary.image("Sample image", x_image)

    with tf.name_scope("Tier1"):
      conv1_layer, conv1_weights, conv1_biases = new_conv_layer(
          input=x_image,
          num_input_channels=num_channels,
          filter_size=filter_size1,
          num_filters=num_filters1,
          use_pooling=True)
      debug("conv1 shape", conv1_layer.shape)

    with tf.name_scope("Tier2"):
      conv2_layer, conv2_weights, conv2_biases = new_conv_layer(
          input=conv1_layer,
          num_input_channels=num_filters1,
          filter_size=filter_size2,
          num_filters=num_filters2,
          use_pooling=True)
      debug("conv2 shape", conv2_layer.shape)

    with tf.name_scope("Tier3"):
      flat_layer, num_features = flatten_layer(conv2_layer)
      debug("flat shape", flat_layer.shape)

    with tf.name_scope("Tier4"):
      fc1_layer, fc1_weights, fc1_biases = new_fc_layer(
          input=flat_layer,
          num_inputs=num_features,
          num_outputs=fc_size,
          use_relu=True)
      debug("fc1_layer shape", fc1_layer.shape)

    with tf.name_scope("Tier5"):
      fc2_layer, fc2_weights, fc2_biases = new_fc_layer(
          input=fc1_layer,
          num_inputs=fc_size,
          num_outputs=num_classes,
          use_relu=False)
      debug("fc2_layer shape", fc2_layer.shape)

    logits = fc2_layer

    return logits, fc1_weights, fc1_biases, fc2_weights, fc2_biases


  def model_fn(features, labels, mode):
    print("In model_fn", labels)
    d_rate = dropout_rate
    if mode in set([tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]):
      d_rate = 0.00000001

    # Build the neural network.
    with tf.name_scope('Model'):
      logits, fc1_weights, fc1_biases, fc2_weights, fc2_biases = (
          build_neural_net(features, d_rate))

      # Predictions.
      pred_classes = tf.argmax(logits, axis=1)

    # In prediction mode we can return the model with predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=pred_classes)

    # Define the loss operation.
    with tf.name_scope('Loss'):
      loss_op = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits,
              labels=labels))

      regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

      # Add the regularization term to the loss.
      loss_op += 5e-4 * regularizers

      tf.summary.scalar("loss", loss_op)

    with tf.name_scope('Accuracy'):
      # Evaluate the accuracy of the model
      accuracy_op = tf.metrics.accuracy(
          labels=labels, predictions=pred_classes)
      tf.summary.scalar("acc0", accuracy_op[0])
      tf.summary.scalar("acc1", accuracy_op[1])

    summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir='logdir',
      summary_op=tf.summary.merge_all())

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #  print("Running predict")
    #  predictions = {
    #    'class_ids': pred_classes[:, tf.newaxis],
    #    'probabilities': tf.nn.softmax(logits),
    #    'logits': logits,
    #  }
    #  return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # If running in eval mode, we can stop here.
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=pred_classes,
          evaluation_hooks=[summary_hook],
          eval_metric_ops={'accuracy': accuracy_op})

    with tf.name_scope('Training'):
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=momentum)
      train_op = optimizer.minimize(
          loss_op, global_step=tf.train.get_global_step())

    # TF Estimators requires to return a EstimatorSpec.
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        training_hooks=[summary_hook],
        eval_metric_ops={'accuracy': accuracy_op})

    return estim_specs

  return tf.estimator.Estimator(model_fn, save_dir)
