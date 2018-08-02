from PIL import Image as PILImage
from PIL import ImageFilter
from PIL import ImageDraw
import random
import argparse
import os


class ImageGenerator:
  def __init__(self, image_size=(64, 64)):
    self.image_size = image_size

  def noise(self):
    return self._create(lambda x: self._random_color())

  def blank(self):
    color = self._random_color()
    return self._create(lambda x: color)

  def circle(self):
    im = self.blank()
    draw = ImageDraw.Draw(im)
    color = self._random_color()
    m = int(random.random() * self.image_size[0] * 0.15)
    w, h = self.image_size
    draw.ellipse((m, m, w-m, h-m), fill=color, outline=color)
    return im


  def _create(self, func=None):
    im = PILImage.new("RGB", self.image_size, (255, 255, 255))
    if func is not None:
      im.putdata(list(map(func, self._flat_zero())))
    return im

  def _random_color(self):
    return (
        int(random.random() * 256),
        int(random.random() * 256),
        int(random.random() * 256)
    )

  def _flat_zero(self):
    return [0] * self.image_size[0] * self.image_size[1]


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Generates noise images")
  parser.add_argument(
      "--out-dir",
      type=str,
      required=True,
      help="The output directory")
  parser.add_argument(
      "--method",
      type=str,
      required=True,
      choices=['blank', 'circle', 'noise'],
      help="The used method")
  parser.add_argument(
      "--random-seed",
      type=int,
      default=None,
      help="The random seed initializer")
  parser.add_argument(
      "-n", "--count",
      dest='count',
      type=int,
      default=50,
      help="Number of instances to generate")

  args = parser.parse_args()
  gen = ImageGenerator()
  generator = getattr(gen, args.method)
  random.seed(args.random_seed)

  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
  for i in range(args.count):
    im = generator()
    filename = os.path.join(args.out_dir, '%s-%03d.png' % (args.method, i))
    im.save(filename)

