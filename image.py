"""
Provides a wrapper class for loading and transforming images.
"""

from common import ROOT_DIR
from functools import lru_cache
from os.path import basename
from os.path import dirname
from PIL import Image as PilImage
from PIL.ImageOps import autocontrast
import hashlib
import numpy as np
import os
import random


class Image:
  """Encapsulates an image sample from the dataset."""

  COUNTER = 0
  """Global counter for transformed images."""

  @staticmethod
  def load(path):
    return Image(
        # TODO: Use some other promise/late API to signify we should read the
        # image.
        image=[path],
        label_name=basename(dirname(path)),
        name=basename(path),
        path=path)

  def __init__(self,
      image,
      name='img',
      label_name='unknown_label',
      path=None,
      filters=[]):
    self.image = image
    self.name = name
    self.label_name = label_name
    self.path = path
    self.filters = filters

  @property
  def basename(self):
    return os.path.basename(self.path)

  @property
  def rel_path(self):
    return self.path[len(ROOT_DIR):].replace('\\', '/')

  @lru_cache(maxsize=None)
  def read(self, image_size, num_channels, monochrome=False):
    """Reads and resizes the image as RGB."""
    if num_channels not in set([1, 3]):
      raise Exception("Supporting only 1 or 3 channel images")

    # Apply filters.
    im = self._apply_filters(self._read())

    # Resize.
    if im.size != image_size:
      im = im.convert('RGBA')
      im.thumbnail(image_size, PilImage.ANTIALIAS)
      background = PilImage.new("RGB", image_size, (255, 255, 255))
      background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
      im = background

    # TODO: Refactor as a filter and on the dataset control level.
    if monochrome:
      im = im.convert('L')
      im = im.point(lambda x: 0 if x < 128 else 255, '1')
      im = im.convert('RGB')

    if num_channels == 1:
      im = im.convert('L')

    return im

  def save(self):
    """Saves the image to its designated path."""
    self._apply_filters(self._read()).save(self.path)

  def remove(self):
    """Deletes the image from disk."""
    os.remove(self.path)

  def transformed(self, func, name_suffix=None):
    """Creates a new transformed image."""
    return self._new(
        image=self.image,
        new_filters=[func],
        name_suffix=name_suffix)

  @lru_cache(maxsize=None)
  def _read(self):
    if isinstance(self.image, (list,)):
      return PilImage.open(self.image[0])
    else:
      return self.image

  def _new(self, image, name_suffix=None, new_filters=None):
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
        filters=self.filters + new_filters,
        label_name=self.label_name)

  def _apply_filters(self, im):
    im = im.convert('RGB')
    for filter_fn in self.filters:
      im = filter_fn(im)
    return im

  def __repr__(self):
    return "Image<%s/%s>" % (self.label_name, self.name)
