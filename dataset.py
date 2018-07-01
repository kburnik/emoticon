from functools import lru_cache
from glob import glob
from image import Image
from label import LabelSet
from os.path import basename
from os.path import dirname
from os.path import exists
from os.path import join
import numpy as np
import random
import shutil
import math
import tensorflow as tf


class DataSetConfig:
  """Encapsulates the configuration of a dataset."""
  def __init__(
      self,
      max_labels=6,
      num_channels=3,
      batch_size=8,
      image_size=(32, 32),
      random_seed=None):
    self.max_labels = max_labels
    self.num_channels = num_channels
    self.batch_size = batch_size
    self.image_size = image_size
    self.random_seed = random_seed


class Data:
  """Memory representation of the data from the data set."""
  def __init__(self, samples, config, name):
    self.samples = samples
    self.config = config
    self.name = name

  @property
  def size(self):
    """Number of data samples."""
    return len(self.samples)

  def shuffled(self):
    """Returns a new Data object with shuffled samples."""
    random.seed(self.config.random_seed)
    new_samples = self.samples[:]
    random.shuffle(new_samples)
    return self._new(new_samples)

  def expanded(self, expander_fn):
    """Returns a new Data object which has an expanded number of samples
       generated by the expander_fn."""
    def expand():
      for sample in self.samples:
        for new_sample in expander_fn(sample):
          yield new_sample
    return self._new(list(expand()))

  def split(self, ratio, names=None):
    """Splits the data in two parts with the provided ratio"""
    split_point = math.ceil(ratio * self.size)
    left = self.samples[:split_point]
    right = self.samples[split_point:]
    if names is None:
      names = ["%.2f" % ratio, "%.2f" % (1 - ratio)]
    return (
        self._new(left, name=self.name + "/" + names[0]),
        self._new(right, name=self.name + "/" + names[1]))

  @lru_cache(maxsize=None)
  def labels(self):
    """Returns all the label indices as an np array"""
    return np.asarray([label.index for image, label in self.samples])

  @lru_cache(maxsize=None)
  def images(self):
    """Returns all the images in an np array."""
    np_data = np.ndarray(
        shape=(
            self.size,
            self.config.image_size[0],
            self.config.image_size[1],
            self.config.num_channels),
        dtype=np.float32)
    for i in range(self.size):
      image = self.samples[i][0].read(
          image_size=self.config.image_size,
          num_channels=self.config.num_channels)
      image_vector = np.asarray(image).reshape(
          *image.size,
          self.config.num_channels)
      # from PIL import Image as PilImage
      # img = PilImage.fromarray(image, 'RGB')
      # img.show()
      np_data[i, ...] = self.normalize(image_vector)
    return np_data

  def input_fn(self):
    """Returns the TF input node for the data."""
    return tf.estimator.inputs.numpy_input_fn(
        x={'x': self.images()},
        y=self.labels(),
        batch_size=self.config.batch_size,
        num_epochs=1,
        shuffle=False)

  def normalize(self, image):
    """Normalizes the image from 0-255 to 0-1 range."""
    return (image.astype(float) / 256.0)

  def _new(self, samples, name=None):
    """Creates a new dataset with provided samples and same config."""
    if name is None:
      name = self.name
    return Data(samples=samples, config=self.config, name=name)


class DataSet:
  """Encapsulates the data set."""
  def __init__(self, path, name="generic", config=None):
    self.path = path
    self.name = name
    self.config = config
    self.label_set = LabelSet(path)

  def print_info(self):
    """Prints information about the dataset"""
    print("Dataset: %s" % self.name)
    print("Path: %s" % self.path)
    print("Labels: %s" % self.label_set.labels())
    print("Samples:")
    for sample in self.get_samples():
      print(sample)

  @lru_cache(maxsize=None)
  def data(self):
    """Returns the Data object of the dataset."""
    return Data(
        samples=list(self.get_samples()),
        config=self.config,
        name=self.name)

  def get_samples(self):
    """Streams all the dataset samples and labels."""
    for label in self.label_set.labels():
      for image_path in glob(join(label.path, "*.png")):
        yield Image.load(image_path), label

  def copy(self, dest, name=None, force=False):
    """Creates a copy of the entire dataset to the destination path."""
    if dirname(dest) != dirname(self.path):
      raise Exception("Destination must match root dir of dataset")
    if exists(dest):
      print("Will be removing directory", dest)
      if not force and input("Type YES to remove: ").strip() != "YES":
        raise Exception("Exiting.")
      shutil.rmtree(dest)
    shutil.copytree(self.path, dest)
    if name is None:
      name = self.name + "-copy"
    return DataSet(dest, config=self.config, name=name)

