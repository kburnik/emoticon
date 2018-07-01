from os.path import basename
from os.path import dirname
from PIL import Image as PilImage
import numpy as np
import random


class Image:
  """Encapsulates an image sample from the dataset."""
  @staticmethod
  def load(path):
    return Image(
        image=PilImage.open(path),
        label_name=basename(dirname(path)),
        name=basename(path))

  def __init__(self, image, name='img', label_name='unknown_label'):
    self.image = image
    self.name = name
    self.label_name = label_name

  @property
  def size(self):
    """Fetches the (width, height) of the image."""
    return self.image.size

  def read(self, image_size, num_channels):
    """Reads and resizes the image as RGB."""
    if num_channels != 3:
      raise Exception("Supporting only 3 channel images")
    im = self.image.convert('RGBA')
    im.thumbnail(image_size, PilImage.ANTIALIAS)
    background = PilImage.new("RGB", image_size, (255, 255, 255))
    background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
    return background

  def transformed(self, func):
    """Creates a new transformed image."""
    return self._new(image=func(self.image))

  def _new(self, image, name_suffix=None):
    """Creates a new, possibly transformed instance."""
    if name_suffix is None:
      name_suffix = "-new%5d" % (random.random() * 10000)
    return Image(
        image,
        name=self.name + name_suffix,
        label_name=self.label_name)

  def __repr__(self):
    return "Image<%s/%s [%dx%d]>" % (
        self.label_name, self.name, self.size[0], self.size[1])
