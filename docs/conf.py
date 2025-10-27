"""Sphinx configuration for aipype documentation."""

import os
import sys
import tomllib
from pathlib import Path

# Add source directories to Python path for autodoc
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "packages" / "aipype" / "src"))
sys.path.insert(0, str(project_root / "packages" / "aipype-extras" / "src"))
sys.path.insert(0, str(project_root / "packages" / "aipype-g" / "src"))

# Extract version from aipype package pyproject.toml
def get_version() -> str:
    """Read version from aipype package pyproject.toml."""
    pyproject_path = project_root / "packages" / "aipype" / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "unknown")
    except (FileNotFoundError, tomllib.TOMLDecodeError, KeyError):
        return "unknown"

# Project information
project = "aipype"
copyright = "2025, aipype contributors"
author = "aipype contributors"
release = get_version()
version = release  # Short version (shown in sidebar)

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Additional MyST settings for better code block handling
myst_heading_anchors = 3  # Generate anchors for h1-h3
myst_update_mathjax = False  # Prevent MathJax conflicts

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True  # Use admonitions for examples
napoleon_use_admonition_for_notes = True    # Use admonitions for notes
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_custom_sections = [('Example', 'params_style')]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Enhanced autodoc settings
autodoc_preserve_defaults = True
autodoc_typehints_format = 'short'
autodoc_class_signature = 'separated'

# Type hints configuration
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
typehints_fully_qualified = False
always_document_param_types = True

# Enable MyST for autodoc docstrings
autodoc_docstring_signature = True

# Autosummary settings
autosummary_generate = False
autosummary_imported_members = False

# Suppress warnings
suppress_warnings = [
    'autosummary.import_cycle',
    'autodoc.import_object'
]

# Clean autodoc settings
autodoc_mock_imports = []
autodoc_member_order = 'bysource'

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Template path
templates_path = ["_templates"]

# List of patterns to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output configuration
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# HTML theme options
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Additional HTML context
html_context = {
    "display_github": True,
    "github_user": "your-org",
    "github_repo": "aipype",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# HTML output settings
html_title = f"{project} {release} documentation"
html_short_title = project
html_favicon = None

# LaTeX output settings (for PDF generation)
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "\\usepackage[Bjornstrup]{fncychap}",
    "printindex": "\\footnotesize\\raggedright\\printindex",
}

latex_documents = [
    ("index", "aipype.tex", "aipype Documentation", "aipype contributors", "manual"),
]

# Manual page output
man_pages = [("index", "aipype", "aipype Documentation", [author], 1)]

# Texinfo output
texinfo_documents = [
    (
        "index",
        "aipype",
        "aipype Documentation",
        author,
        "aipype",
        "Modular AI agent framework with declarative pipeline-based task orchestration.",
        "Miscellaneous",
    ),
]

# Extension settings
add_module_names = False
python_use_unqualified_type_names = True


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip specific members from documentation."""
    # Skip URLFetcher class attributes that we don't want documented
    if name in ("DEFAULT_HEADERS", "DEFAULT_USER_AGENT"):
        return True
    return skip


def setup(app):
    """Setup function for Sphinx."""
    app.connect("autodoc-skip-member", autodoc_skip_member)