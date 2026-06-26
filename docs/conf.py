from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "SDA Simulator V2"
copyright = "2026"
author = "SDA Simulator V2 contributors"

try:
    release = version("sda-simulator-v2")
except PackageNotFoundError:
    release = "0.1.0"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints = "description"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"
html_title = "SDA Simulator V2"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
