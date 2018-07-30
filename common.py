"""
Provides common functions and constants used in the project.
"""
import os


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
"""The project root directory."""

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data")
"""The data root directory"""

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, ".model")
"""Directory where the models are cached."""

REPORT_DIR = os.path.join(ROOT_DIR, "reports")
"""Directory for storing prediction reports."""


def data_dir(dir_name):
  return os.path.join(DATA_ROOT_DIR, dir_name)


class DataPath:
  ALL = data_dir("all")
  """All images from the scraped source."""

  DUMMY = data_dir("dummy")
  """Picked out images for sanity checking - debugging."""

  SIMPLE = data_dir("simple")
  """Picked out images for a simple data set - debugging."""

  COMMON_3 = data_dir("common-3")
  """Picked out images for common emoticons."""

  COMMON_3_GRAYSCALE = data_dir("common-3-grayscale")
  """Picked out images for common emoticons (grayscale)."""

  COMMON_3_EXTENDED = data_dir("common-3-extended")
  """Merged images from emojipedia.org and unicode.org across 3 common types."""

  COMMON_4 = data_dir("common-4")
  """Picked out images for common emoticons."""

  COMMON_6 = data_dir("common-6")
  """Picked out images for common emoticons."""

  COMMON_6_EXTENDED = data_dir("common-6-extended")
  """Merged images from emojipedia.org and unicode.org across 6 common types."""

  COMMON_6_EXTRA = data_dir("common-6-extra")
  """
  Merged images from emojipedia.org, unicode.org and Google search across 6
  common types.
  """

  COMMON_6_GRAYSCALE = data_dir("common-6-grayscale")
  """Picked out images for common emoticons (grayscale)."""

  GENERATED = data_dir("generated")
  """Generated emoticons."""

  GENERATED_EXPANDED_20 = data_dir("generated-expanded-20")
  """Generated emoticons with elastic expansion and scaled down to 64x64."""

  @staticmethod
  def names():
    return [
      name for name in dir(DataPath)
      if not name.startswith('__') and name.upper() == name]

  @staticmethod
  def get(name):
    return getattr(DataPath, name.upper())

