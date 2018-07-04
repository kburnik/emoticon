"""
Provides common functions and constants used in the project.
"""
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
"""The project root directory."""

DATA_ROOT_DIR = os.path.join(ROOT_DIR, "data")
"""The data root directory"""

DATA_ALL_DIR = os.path.join(DATA_ROOT_DIR, "all")
"""All images from the scraped source."""

DATA_DUMMY_DIR = os.path.join(DATA_ROOT_DIR, "dummy")
"""Picked out images for sanity checking - debugging."""

DATA_SIMPLE_DIR = os.path.join(DATA_ROOT_DIR, "simple")
"""Picked out images for a simple data set - debugging."""

DATA_COMMON_3_DIR = os.path.join(DATA_ROOT_DIR, "common-3")
"""Picked out images for common emoticons."""

DATA_COMMON_4_DIR = os.path.join(DATA_ROOT_DIR, "common-4")
"""Picked out images for common emoticons."""

DATA_COMMON_6_DIR = os.path.join(DATA_ROOT_DIR, "common-6")
"""Picked out images for common emoticons."""

DATA_COMMON_6_GRAYSCALE_DIR = os.path.join(DATA_ROOT_DIR, "common-6-grayscale")
"""Picked out images for common emoticons (grayscale)."""

DATA_GENERATED_DIR = os.path.join(DATA_ROOT_DIR, "generated")
"""Generated emoticons."""

MODEL_SAVE_DIR = os.path.join(ROOT_DIR, ".model")
"""Directory where the models are cached."""
