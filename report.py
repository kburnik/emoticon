from common import REPORT_DIR
from glob import glob
import json
import os

class Reports:
  """Encapsulates generated prediction reports."""
  @staticmethod
  def paths():
    """Returns all the report paths."""
    return glob(os.path.join(REPORT_DIR, '*.json'))

  @staticmethod
  def basenames():
    """Returns all the report basenames."""
    return list(map(os.path.basename, Reports.paths()))

  @staticmethod
  def names():
    """Returns all the report names."""
    return list(map(lambda bn: bn.replace('.json', ''), Reports.basenames()))

  @staticmethod
  def load_by_name(name):
    """Returns a decode report by its name."""
    filename = os.path.join(REPORT_DIR, name + '.json')
    with open(filename, 'r') as f:
      return json.load(f)
