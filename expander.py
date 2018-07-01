import random
from PIL import Image as PilImage


def rand_offset(size):
  return round(random.random() * size * 2 - size)


def translate(image):
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
  rot = image.rotate(rand_offset(25), expand=1)
  fff = PilImage.new('RGBA', rot.size, (255,)*4)
  return PilImage.composite(rot, fff, rot)


def expander(sample, factor):
  image, label = sample

  for i in range(factor):
    new_image = image.transformed(translate)
    # new_image.image.show()
    # input("next? ")
    yield (new_image, label)

