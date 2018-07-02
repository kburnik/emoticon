"""
Provides common functions and constants used in the project.
"""
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
"""The project root directory."""

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data")
"""The data root directory"""

DATA_RAW_DIR = os.path.join(DATA_ROOT_DIR, "raw")
"""Raw images with many categories."""

DATA_DUMMY_DIR = os.path.join(DATA_ROOT_DIR, "dummy")
"""Picked out images for sanity checking - debugging."""

DATA_SIMPLE_DIR = os.path.join(DATA_ROOT_DIR, "simple")
"""Picked out images for a simple data set - debugging."""

DATA_COMMON_6_DIR = os.path.join(DATA_ROOT_DIR, "common-6")
"""Picked out images for common emoticons."""

DATA_COMMON_6_GRAYSCALE_DIR = os.path.join(DATA_ROOT_DIR, "common-6-grayscale")
"""Picked out images for common emoticons (grayscale)."""

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, ".model")
"""Directory where the models are cached."""
