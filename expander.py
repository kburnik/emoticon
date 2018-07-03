"""
Provides the methods for artificially expanding a dataset by generating
variations of source images. This is useful as a regularization technique
when not a lot of data is available (i.e. should prevent overfitting).
"""

from PIL import Image as PilImage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def do_elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size])
    pts2 = pts1 + random_state.uniform(
        -alpha_affine,
        alpha_affine,
        size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(
        np.arange(shape[1]),
        np.arange(shape[0]),
        np.arange(shape[2]))

    indices = (
        np.reshape(y+dy, (-1, 1)),
        np.reshape(x+dx, (-1, 1)),
        np.reshape(z, (-1, 1)))

    im = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return im

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


def affine_transform(image):
  """Applies random affine transformation to an image."""
  return rotate(translate(image))


def pil_to_opencv_image(pil_image):
  open_cv_image = np.array(pil_image)
  # Convert RGB to BGR
  open_cv_image = open_cv_image[:, :, ::-1].copy()
  return open_cv_image


def opencv_to_pil_image(open_cv_image):
  open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
  pil_im = PilImage.fromarray(open_cv_image)
  return pil_im


def elastic_transform(image):
  """Applies elastic transformation to an image."""
  im = pil_to_opencv_image(image)
  im = do_elastic_transform(
      im,
      alpha=im.shape[1] * 0.75,
      sigma=im.shape[1] * 0.04,
      alpha_affine=im.shape[1] * 0.025)
  return opencv_to_pil_image(im)


def expander(sample, factor):
  """Generates variations of a sample by transforming the image."""
  image, label = sample

  for i in range(factor):
    new_image = image.transformed(elastic_transform)
    # new_image.image.show()
    # input("next? ")
    yield (new_image, label)


if __name__ == '__main__':
  elastic_transform(PilImage.open("./grid.png")).show()
  elastic_transform(PilImage.open("./data/common-6/crying-face/00.png")).show()
