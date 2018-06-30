from os.path import basename
from os.path import dirname
from PIL import Image as PilImage
import numpy as np

class Image:
  """Encapsulates an image sample from the dataset."""
  def __init__(self, path):
    self.path = path
    """Path to the image."""

    self.label_name = basename(dirname(path))
    """Name of the label the image belongs to."""

    self.name = basename(path)
    """Basename of the image."""

  @property
  def size(self):
    """Fetches the (width, height) of the image."""
    return PilImage.open(self.path).size

  def read(self, image_size, num_channels):
    """Reads and resizes the image as RGB into an np array."""
    if num_channels != 3:
      raise Exception("Supporting only 3 channel images")
    im = PilImage.open(self.path).convert('RGBA')
    im.thumbnail(image_size, PilImage.ANTIALIAS)

    background = PilImage.new("RGB", image_size, (255, 255, 255))
    background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
    return np.asarray(background).reshape(*image_size, 3)

  def __repr__(self):
    return "Image<%s/%s [%dx%d]>" % (
        self.label_name, self.name, self.size[0], self.size[1])
