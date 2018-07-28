"""
Provides classes for working with data set labels.
"""
from glob import glob
from os.path import join
from os.path import basename

class Label:
  """Encapsulates a label."""
  def __init__(self, path, index):
    self.path = path
    self.name = basename(path)
    self.index = index

  def __repr__(self):
    return "Label<%s:%d>" % (self.name, self.index)

class LabelSet:
  """Encapsulates dataset labels."""
  def __init__(self, path):
    self.path = path

  @property
  def size(self):
    """Number of distinct labels."""
    return len(self.labels())

  def labels(self):
    """Returns the list of Label objects."""
    return list(self._enumerate())

  def _enumerate(self):
    for i, label_dir in enumerate(sorted(glob(join(self.path, "*")))):
      yield Label(path=label_dir, index=i)
