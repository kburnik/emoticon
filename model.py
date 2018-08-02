"""
Specifies the machine learning model (a convolutional neural network) and
provides options for building variations.
"""

from math import floor
from math import sqrt
from visual import get_conv_output_image
from visual import put_kernels_on_grid
import tensorflow as tf


class Model:
  """Encapsulates a built model."""
  def __init__(self, convolutional, fully_connected):
    self.convolutional = convolutional
    self.fully_connected = fully_connected
    self.logits = fully_connected[-1].layer

  def compute_loss(self, labels):
    """Builds and returns the loss operation for the model."""
    with tf.name_scope('Loss'):
      conv_loss = self._compute_hidden_stack_loss(
            self.convolutional,
            prefix='conv')

      fc_loss = self._compute_hidden_stack_loss(
            self.fully_connected,
            prefix='fc')

      logits_loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=self.logits,
              labels=labels))

      # Add the regularization terms to the loss (from previous layers).
      loss_op = 0.25 * conv_loss + 0.5 * fc_loss + logits_loss
      tf.summary.scalar("logits", logits_loss)
      tf.summary.scalar("total", loss_op)
      return loss_op

  def _compute_hidden_stack_loss(self, layers, prefix):
    """Computes the loss for weights and biases of provided layers"""
    terms = tf.Variable(0, name=prefix + '_reg_term', dtype=tf.float32)
    for i, layer in enumerate(layers):
      weights_loss = tf.nn.l2_loss(layer.weights)
      biases_loss = tf.nn.l2_loss(layer.biases)
      tf.summary.scalar("%s%d_weights" % (prefix, (i + 1)), weights_loss)
      tf.summary.scalar("%s%d_biases" % (prefix, (i + 1)), biases_loss)
      terms.assign_add(weights_loss + biases_loss)
    tf.summary.scalar("%s_regularizers" % prefix, terms)
    return terms


class Layer:
  """Encapsulates a model layer."""
  def __init__(self, layer, weights, biases, name):
    self.layer = layer
    self.weights = weights
    self.biases = biases
    self.name = name


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
    random_seed=None,
    use_relu=False,
    name=None):
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
  if use_relu:
    layer = tf.nn.leaky_relu(layer)

  if use_dropout:
    layer = tf.nn.dropout(layer, dropout_rate, seed=random_seed)

  return Layer(layer=layer, weights=weights, biases=biases, name=name)


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
    random_seed=None,
    name=None):
  """Creates a fully connected layer with ReLU activation."""

  weights = new_weights(shape=[num_inputs, num_outputs])
  biases = new_biases(length=num_outputs)

  layer = tf.matmul(input, weights) + biases

  if use_relu:
    layer = tf.nn.leaky_relu(layer)

  if use_dropout:
    layer = tf.nn.dropout(
        layer,
        dropout_rate,
        seed=random_seed,
        name="dropout")

  return Layer(layer=layer, weights=weights, biases=biases, name=name)


def build_model(
    ds_config,
    num_classes,
    args,
    debug=noop):
  """Builds the CNN for training, evaluation or prediction."""
  num_channels = ds_config.num_channels
  image_size = ds_config.image_size
  random_seed = ds_config.random_seed

  def build_neural_net(features, mode):
    train_mode = mode == tf.estimator.ModeKeys.TRAIN

    # Convolutional layer 1.
    filter_size1 = args.filter_size
    num_filters1 = 25
    num_rows1 = floor(sqrt(num_filters1))
    filter_stride1 = 1
    pool_stride1 = 2

    # Convolutional layer 2.
    filter_size2 = args.filter_size
    num_filters2 = 36
    num_rows2 = floor(sqrt(num_filters2))
    filter_stride2 = 1
    pool_stride2 = 2

    # Fully-connected layers.
    fc1_size = num_classes * 18
    fc2_size = num_classes * 12

    data = features['x']

    with tf.name_scope("Input"):
      x_image = tf.reshape(
          data,
          [-1, image_size[1], image_size[0], num_channels])
      tf.summary.image("Sample image", x_image)

    with tf.name_scope("Convolutional-1"):
      conv1 = new_conv_layer(
          input=x_image,
          num_input_channels=num_channels,
          filter_size=filter_size1,
          num_filters=num_filters1,
          filter_stride=filter_stride1,
          pool_stride=pool_stride1,
          use_pooling=(args.pooling & 1) > 0,
          use_relu=(args.relu & 1) > 0,
          use_dropout=(args.dropout & 1) > 0 and train_mode,
          random_seed=random_seed,
          name='conv1')
      debug("conv1 shape", conv1.layer.shape)
      tf.summary.image(
          'conv1/kernels', put_kernels_on_grid(conv1.weights), max_outputs=1)
      tf.summary.image(
          'conv1/out',
          get_conv_output_image(conv1.layer, num_filters1, rows=num_rows1))

    with tf.name_scope("Convolutional-2"):
      conv2 = new_conv_layer(
          input=conv1.layer,
          num_input_channels=num_filters1,
          filter_size=filter_size2,
          num_filters=num_filters2,
          filter_stride=filter_stride2,
          pool_stride=pool_stride2,
          use_pooling=(args.pooling & 2) > 0,
          use_relu=(args.relu & 2) > 0,
          use_dropout=(args.dropout & 2) > 0 and train_mode,
          random_seed=random_seed,
          name='conv2')
      debug("conv2 shape", conv2.layer.shape)
      tf.summary.image(
          'conv2/out',
          get_conv_output_image(conv2.layer, num_filters2, rows=num_rows2))

    with tf.name_scope("Flat-Tier"):
      flat_layer, num_features = flatten_layer(conv2.layer)
      debug("flat shape", flat_layer.shape)

    with tf.name_scope("Fully-connected-1"):
      fc1 = new_fc_layer(
          input=flat_layer,
          num_inputs=num_features,
          num_outputs=fc1_size,
          use_relu=(args.relu & 4) > 0,
          use_dropout=(args.dropout & 4) > 0 and train_mode,
          dropout_rate=args.dropout_rate,
          name='fc1')
      debug("fc1_layer shape", fc1.layer.shape)

    with tf.name_scope("Fully-connected-2"):
      fc2 = new_fc_layer(
          input=fc1.layer,
          num_inputs=fc1_size,
          num_outputs=fc2_size,
          use_relu=(args.relu & 8) > 0,
          use_dropout=(args.dropout & 8) > 0 and train_mode,
          name='fc2')
      debug("fc2_layer shape", fc2.layer.shape)

    with tf.name_scope("Fully-connected-3"):
      fc3 = new_fc_layer(
          input=fc2.layer,
          num_inputs=fc2_size,
          num_outputs=num_classes,
          use_relu=False,
          use_dropout=(args.dropout & 16) > 0 and train_mode,
          name='fc3')
      debug("fc3_layer shape", fc3.layer.shape)

    return Model(
        convolutional=[conv1, conv2],
        fully_connected=[fc1, fc2, fc3])

  def model_fn(features, labels, mode):
    # Build the neural network.
    with tf.name_scope('Model'):
      model = build_neural_net(features, mode)

      # Predictions.
      pred_classes = tf.argmax(model.logits, axis=1)

    # In prediction mode we can return the model with predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=pred_classes)

    # Define the loss operation.
    loss_op = model.compute_loss(labels)

    with tf.name_scope('Accuracy'):
      # Evaluate the accuracy of the model
      accuracy_op = tf.metrics.accuracy(
          labels=labels, predictions=pred_classes)
      tf.summary.scalar("acc0", accuracy_op[0])
      tf.summary.scalar("acc1", accuracy_op[1])

    summary_hook = tf.train.SummarySaverHook(
        save_steps=args.training_steps,
        output_dir=args.log_dir,
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
          learning_rate=args.learning_rate,
          momentum=args.momentum)
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

  return tf.estimator.Estimator(model_fn, args.model_dir)
