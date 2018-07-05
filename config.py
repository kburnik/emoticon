import argparse
from common import DataPath
import sys


def parse_config(description="Run operations on the model"):
  parser = argparse.ArgumentParser(
      description=description)

  parser.add_argument(
      "--dataset",
      type=str,
      default='COMMON_3',
      choices=DataPath.names(),
      help="Data set to use")
  parser.add_argument(
      "--num-channels",
      type=int,
      default=3,
      help="Number of channels in each image (e.g. RGB = 3)")
  parser.add_argument(
      "--batch-size",
      type=int,
      default=20,
      help="The training batch size")
  parser.add_argument(
      "--expansion-factor",
      type=int,
      default=0,
      help="The factor for expanding the dataset in memory (0 = no expansion)")
  parser.add_argument(
      "--image-size",
      type=int,
      default=64,
      help="The target width/height of each image as input to the model")
  parser.add_argument(
      "--random-seed",
      type=int,
      default=271941,
      help="The random seed for splitting the data and other random state")
  parser.add_argument(
      "--split-ratio",
      type=float,
      default=0.6,
      help="The train/test split ratio")
  parser.add_argument(
      "--use-dropout",
      type=bool,
      default=True,
      help="Whether to use dropout during training")
  parser.add_argument(
      "--dropout-rate",
      type=float,
      default=0.5,
      help="The dropout probability in the model")
  parser.add_argument(
      "--learning-rate",
      type=float,
      default=0.5,
      help="The training learning rate for MomentumOptimizer")
  parser.add_argument(
      "--momentum",
      type=float,
      default=0.01,
      help="The training momentum for MomentumOptimizer")
  parser.add_argument(
      "--training-steps",
      type=int,
      default=100,
      help="Number of steps in each training iteration")
  parser.add_argument(
      "--show-data",
      type=bool,
      default=False,
      help="Whether to show samples of the data before training")
  return parser.parse_args()


class Unbuffered(object):
  def __init__(self, stream):
      self.stream = stream
  def write(self, data):
      self.stream.write(data)
      self.stream.flush()
  def writelines(self, datas):
      self.stream.writelines(datas)
      self.stream.flush()
  def __getattr__(self, attr):
      return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)
