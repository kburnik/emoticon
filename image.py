"""
Provides a wrapper class for loading and transforming images.
"""

from os.path import basename
from os.path import dirname
from PIL import Image as PilImage
from PIL.ImageOps import autocontrast
import numpy as np
import random
import os


class Image:
  """Encapsulates an image sample from the dataset."""

  COUNTER = 0
  """Global counter for transformed images."""

  @staticmethod
  def load(path):
    return Image(
        image=PilImage.open(path),
        label_name=basename(dirname(path)),
        name=basename(path),
        path=path)

  def __init__(self, image, name='img', label_name='unknown_label', path=None):
    self.image = image
    self.name = name
    self.label_name = label_name
    self.path = path

  @property
  def size(self):
    """Fetches the (width, height) of the image."""
    return self.image.size

  def read(self, image_size, num_channels):
    """Reads and resizes the image as RGB."""
    if num_channels not in set([1, 3]):
      raise Exception("Supporting only 1 or 3 channel images")
    im = self.image.convert('RGBA')
    im.thumbnail(image_size, PilImage.ANTIALIAS)
    background = PilImage.new("RGB", image_size, (255, 255, 255))
    background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
    if num_channels == 1:
      background = background.convert('L')
    return background

  def save(self):
    """Saves the image to its designated path."""
    self.image.save(self.path)

  def remove(self):
    """Deletes the image from disk."""
    os.remove(self.path)

  def transformed(self, func, name_suffix=None):
    """Creates a new transformed image."""
    return self._new(
        image=func(self.image.convert('RGB')),
        name_suffix=name_suffix)

  def _new(self, image, name_suffix=None):
    """Creates a new, possibly transformed instance."""
    if name_suffix is None:
      name_suffix = "-%06d" % Image.COUNTER
      Image.COUNTER += 1

    new_name = self.name.replace('.png', name_suffix) + ".png"
    new_path = self.path.replace('.png', name_suffix) + ".png"

    return Image(
        image,
        name=new_name,
        path=new_path,
        label_name=self.label_name)

  def __repr__(self):
    return "Image<%s/%s [%dx%d]>" % (
        self.label_name, self.name, self.size[0], self.size[1])
