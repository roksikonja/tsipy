import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "tsipy"
copyright = "2021, Rok Šikonja"
author = "Rok Šikonja"

version = "1.0"
release = "1.0.2"

pygments_style = "sphinx"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
    "nbsphinx",
    "nbsphinx_link",
]

if "spelling" in sys.argv:
    extensions.append("sphinxcontrib.spelling")

spelling_lang = "en_US"
spelling_word_list_filename = "spelling_wordlist.txt"

bibtex_bibfiles = ["references.bib"]
