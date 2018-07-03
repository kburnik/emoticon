"""
Specifies the machine learning model (a convolutional neural network) and
provides options for building variations.
"""

import tensorflow as tf
from common import MODEL_SAVE_DIR

def noop(*args):
  """Stand-in method for handling debug info."""
  pass


def weight_variable(shape, name):
  """Creates a new weight variable with common defaults."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)


def new_weights(shape):
  """Creates a set of weight values with common defaults."""
  return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
  """Creates a set of bias values with common defaults."""
  return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(
    input,
    num_input_channels,
    filter_size,
    num_filters,
    use_pooling=True,
    filter_stride=1,
    pool_stride=1,
    use_dropout=False,
    dropout_rate=0.5,
    random_seed=None):
  """Creates a new ReLU convolutional layer."""
  shape = [filter_size, filter_size, num_input_channels, num_filters]
  weights = new_weights(shape=shape)
  biases = new_biases(length=num_filters)

  layer = tf.nn.conv2d(
      input=input,
      filter=weights,
      strides=[1, filter_stride, filter_stride, 1],
      padding='SAME')

  layer += biases

  if use_pooling:
    layer = tf.nn.max_pool(
        value=layer,
        ksize=[1, pool_stride, pool_stride, 1],
        strides=[1, pool_stride, pool_stride, 1],
        padding='SAME')

  # Rectified Linear Unit (ReLU).
  layer = tf.nn.relu(layer)

  if use_dropout:
    layer = tf.nn.dropout(layer, dropout_rate, seed=random_seed)

  return layer, weights, biases


def flatten_layer(layer):
  """Flattens a previous multi-dim output to a single vector shape."""
  layer_shape = layer.get_shape()
  num_features = layer_shape[1:4].num_elements()
  flat_layer = tf.reshape(layer, [-1, num_features])
  return flat_layer, num_features


def new_fc_layer(
    input,
    num_inputs,
    num_outputs,
    use_relu=True,
    use_dropout=False,
    dropout_rate=0.5,
    random_seed=None):
  """Creates a fully connected layer with ReLU activation."""

  weights = new_weights(shape=[num_inputs, num_outputs])
  biases = new_biases(length=num_outputs)

  layer = tf.matmul(input, weights) + biases

  if use_relu:
    layer = tf.nn.relu(layer)

  if use_dropout:
    layer = tf.nn.dropout(
        layer,
        dropout_rate,
        seed=random_seed,
        name="dropout")

  return layer, weights, biases


def build_model(
    ds_config,
    num_classes,
    learning_rate=0.1,
    momentum=0.9,
    use_pooling=True,
    use_dropout=False,
    dropout_rate=0.5,
    debug=noop,
    save_dir=MODEL_SAVE_DIR):
  """Builds the CNN for training, evaluation or prediction."""
  num_channels = ds_config.num_channels
  image_size = ds_config.image_size
  random_seed = ds_config.random_seed

  def build_neural_net(features):
    # Convolutional layer 1.
    filter_size1 = 8
    num_filters1 = 16
    filter_stride1 = 1
    pool_stride1 = 2

    # Convolutional layer 2.
    filter_size2 = 8
    num_filters2 = 36
    filter_stride2 = 1
    pool_stride2 = 2

    # Fully-connected layers.
    fc1_size = num_classes * 16

    data = features['x']

    with tf.name_scope("Input"):
      x_image = tf.reshape(
          data,
          [-1, image_size[1], image_size[0], num_channels])
      tf.summary.image("Sample image", x_image)

    with tf.name_scope("Convolutional-1"):
      conv1_layer, conv1_weights, conv1_biases = new_conv_layer(
          input=x_image,
          num_input_channels=num_channels,
          filter_size=filter_size1,
          num_filters=num_filters1,
          use_pooling=use_pooling,
          filter_stride=filter_stride1,
          pool_stride=pool_stride1,
          use_dropout=False,
          random_seed=random_seed)
      debug("conv1 shape", conv1_layer.shape)

    with tf.name_scope("Convolutional-2"):
      conv2_layer, conv2_weights, conv2_biases = new_conv_layer(
          input=conv1_layer,
          num_input_channels=num_filters1,
          filter_size=filter_size2,
          num_filters=num_filters2,
          use_pooling=use_pooling,
          filter_stride=filter_stride2,
          pool_stride=pool_stride2,
          use_dropout=False,
          random_seed=random_seed)
      debug("conv2 shape", conv2_layer.shape)

    with tf.name_scope("Flat-Tier"):
      flat_layer, num_features = flatten_layer(conv2_layer)
      debug("flat shape", flat_layer.shape)

    with tf.name_scope("Fully-connected-1"):
      fc1_layer, fc1_weights, fc1_biases = new_fc_layer(
          input=flat_layer,
          num_inputs=num_features,
          num_outputs=fc1_size,
          use_relu=True,
          use_dropout=True,
          dropout_rate=dropout_rate)
      debug("fc1_layer shape", fc1_layer.shape)

    with tf.name_scope("Fully-connected-2"):
      fc2_layer, fc2_weights, fc2_biases = new_fc_layer(
          input=fc1_layer,
          num_inputs=fc1_size,
          num_outputs=num_classes,
          use_relu=False,
          use_dropout=False)
      debug("fc2_layer shape", fc2_layer.shape)

    logits = fc2_layer

    return (
        conv1_weights, conv1_biases,
        conv2_weights, conv2_biases,
        fc1_weights, fc1_biases,
        fc2_weights, fc2_biases,
        logits,
        )


  def model_fn(features, labels, mode):
    # Build the neural network.
    with tf.name_scope('Model'):
      conv1_weights, conv1_biases, \
          conv2_weights, conv2_biases, \
          fc1_weights, fc1_biases, \
          fc2_weights, fc2_biases, \
          logits = build_neural_net(features)

      # Predictions.
      pred_classes = tf.argmax(logits, axis=1)

    # In prediction mode we can return the model with predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=pred_classes)

    # Define the loss operation.
    with tf.name_scope('Loss'):
      logits_loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits,
              labels=labels))

      conv1_weights_loss = tf.nn.l2_loss(conv1_weights)
      conv1_biases_loss = tf.nn.l2_loss(conv1_biases)
      conv2_weights_loss = tf.nn.l2_loss(conv2_weights)
      conv2_biases_loss = tf.nn.l2_loss(conv2_biases)
      conv_regularizers = (
          conv1_weights_loss + conv1_biases_loss +
          conv2_weights_loss + conv2_biases_loss)

      fc1_weights_loss = tf.nn.l2_loss(fc1_weights)
      fc1_biases_loss = tf.nn.l2_loss(fc1_biases)
      fc2_weights_loss = tf.nn.l2_loss(fc2_weights)
      fc2_biases_loss = tf.nn.l2_loss(fc2_biases)
      fc_regularizers = (
          fc1_weights_loss + fc1_biases_loss +
          fc2_weights_loss + fc2_biases_loss)

      # Add the regularization terms to the loss.
      loss_op = logits_loss + 0.1 * conv_regularizers + 0.1 * fc_regularizers

      # Export scalars for loss.
      tf.summary.scalar("conv1_weights", conv1_weights_loss)
      tf.summary.scalar("conv1_biases", conv1_biases_loss)
      tf.summary.scalar("conv2_weights", conv1_weights_loss)
      tf.summary.scalar("conv2_biases", conv1_biases_loss)
      tf.summary.scalar("conv_regularizers", conv_regularizers)

      tf.summary.scalar("fc1_weights", fc1_weights_loss)
      tf.summary.scalar("fc1_biases", fc1_biases_loss)
      tf.summary.scalar("fc2_weights", fc2_weights_loss)
      tf.summary.scalar("fc2_biases", fc2_biases_loss)
      tf.summary.scalar("fc_regularizers", fc_regularizers)

      tf.summary.scalar("logits", logits_loss)
      tf.summary.scalar("total", loss_op)

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

    # If running in eval mode, we can stop here.
    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=pred_classes,
          loss=loss_op,
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
