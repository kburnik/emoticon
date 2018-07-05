"""
Provides common functions and constants used in the project.
"""
import os


def data_dir(dir_name):
  return os.path.join(DATA_ROOT_DIR, dir_name)


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
"""The project root directory."""

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data")
"""The data root directory"""

DATA_ALL_DIR = data_dir("all")
"""All images from the scraped source."""

DATA_DUMMY_DIR = data_dir("dummy")
"""Picked out images for sanity checking - debugging."""

DATA_SIMPLE_DIR = data_dir("simple")
"""Picked out images for a simple data set - debugging."""

DATA_COMMON_3_DIR = data_dir("common-3")
"""Picked out images for common emoticons."""

DATA_COMMON_3_GRAYSCALE_DIR = data_dir("common-3-grayscale")
"""Picked out images for common emoticons (grayscale)."""

DATA_COMMON_4_DIR = data_dir("common-4")
"""Picked out images for common emoticons."""

DATA_COMMON_6_DIR = data_dir("common-6")
"""Picked out images for common emoticons."""

DATA_COMMON_6_GRAYSCALE_DIR = data_dir("common-6-grayscale")
"""Picked out images for common emoticons (grayscale)."""

DATA_GENERATED_DIR = data_dir("generated")
"""Generated emoticons."""

DATA_GENERATED_EXPANDED_20_DIR = data_dir("generated-expanded-20")
"""Generated emoticons with elastic expansion and scaled down to 64x64."""

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, ".model")
"""Directory where the models are cached."""
