"""
Provides the methods for artificially expanding a dataset by generating
variations of source images. This is useful as a regularization technique
when not a lot of data is available (i.e. should prevent overfitting).
"""

import random
from PIL import Image as PilImage


def rand_offset(size):
  """Generates a random offset in the interval [-size, size]."""
  return round(random.random() * size * 2 - size)


def translate(image):
  """Randomly translates an image."""
  return image.convert('RGB').transform(
      image.size,
      PilImage.AFFINE,
      (
        1,
        0,
        rand_offset(5),
        0,
        1,
        rand_offset(5)
      ),
      fill=1,
      fillcolor="white")


def rotate(image):
  """Randomly rotates an image."""
  rot = image.convert('RGBA').rotate(rand_offset(15), expand=1)
  fff = PilImage.new('RGBA', rot.size, (255,)*4)
  return PilImage.composite(rot, fff, rot)


def expander(sample, factor):
  """Generates variations of a sample by transforming the image."""
  image, label = sample

  for i in range(factor):
    new_image = image.transformed(translate).transformed(rotate)
    # new_image.image.show()
    # input("next? ")
    yield (new_image, label)

