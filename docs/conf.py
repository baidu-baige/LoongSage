"""Sphinx configuration shared by the docs/en and docs/zh source trees.

Both are built through docs/build.sh with this directory as the configuration
directory, so every path below resolves relative to this file.
"""

import os
import pathlib
import re

# -- Which tree is being built -------------------------------------------------

# "en" | "zh", set by build.sh. Also the URL segment of the published output.
DOC_LANG = os.environ.get("DOC_LANG", "en")
if DOC_LANG not in ("en", "zh"):
    raise ValueError(f"DOC_LANG must be 'en' or 'zh', got {DOC_LANG!r}")

# Locale for the theme's UI strings; the published URL segment stays /zh/.
language = {"en": "en", "zh": "zh_CN"}[DOC_LANG]

# -- Project -------------------------------------------------------------------

project = "LoongSage"
author = "LoongSage Team"
copyright = "2026, LoongSage Team"

html_title = {"en": "LoongSage Documentation", "zh": "LoongSage 文档"}[DOC_LANG]
html_short_title = "LoongSage"

# -- General -------------------------------------------------------------------

extensions = [
    "myst_parser",             # lets Sphinx read Markdown
    "sphinx_copybutton",       # copy button on code blocks
    "sphinx.ext.mathjax",      # $$ ... $$ rendering
    "sphinx.ext.intersphinx",  # cross-project refs (no mappings yet)
]

root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

UNPUBLISHED_PAGES = [
    "controller.md",
    "agentflow-framework.md",
    "resource-scheduler.md",
    "transfer-mesh.md",
]

exclude_patterns = ["build", "Thumbs.db", ".DS_Store", *UNPUBLISHED_PAGES]

intersphinx_mapping: dict[str, tuple[str, None]] = {}

myst_enable_extensions = [
    "dollarmath",  # $$ math
    "amsmath",
    "deflist",
    "colon_fence",  # ::: fences
]
myst_heading_anchors = 3  # anchors for in-page "#heading" links

# -- HTML ----------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/lang-toggle.js"]
html_logo = "_static/image/logo.png"
html_last_updated_fmt = "%Y-%m-%d"

copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

html_theme_options = {
    "show_navbar_depth": 2,     # sidebar: expand two levels by default
    "show_toc_level": 2,        # right-hand page TOC: h2 + h3
    "home_page_in_toc": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
}

# -- Repository buttons --------------------------------------------------------

REPO_URL = "https://github.com/baidu-baige/LoongSage"
REPO_BRANCH = "main"

html_theme_options.update(
    {
        "repository_url": REPO_URL,
        "repository_branch": REPO_BRANCH,
        # Includes the language segment: the pages live in docs/<lang>/.
        "path_to_docs": f"docs/{DOC_LANG}",
        "use_repository_button": True,
        "use_source_button": True,
        "use_edit_page_button": True,
        "use_issues_button": True,
    }
)

# Canonical links. The published layout puts English at the site root and
# Chinese under /zh/ (see docs/preview.sh), so each tree gets its own base URL:
# a single site-wide value would make every Chinese page declare the English
# page as its canonical one.
SITE_URL = "https://baidu-baige.github.io/LoongSage/"
html_baseurl = SITE_URL if DOC_LANG == "en" else f"{SITE_URL}{DOC_LANG}/"

# -- Links to source code ------------------------------------------------------

# Code links in the Markdown are repository-relative, so that they also work
# when the files are read on GitHub; point them at the repository at read time,
# which also keeps Sphinx from turning them into file downloads. GitHub serves
# files under /blob/ and directories under /tree/.
CODE_BLOB_URL = f"{REPO_URL}/blob/{REPO_BRANCH}"
CODE_TREE_URL = f"{REPO_URL}/tree/{REPO_BRANCH}"

_DOCS_DIR = pathlib.Path(__file__).parent.resolve()
_REPO_ROOT = _DOCS_DIR.parent
_MD_LINK_RE = re.compile(r"(?<=]\()\s*(?!\w+:|#)([^)\s]+)")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def _code_url(md_file: pathlib.Path, target: str) -> str | None:
    """Repository URL for a link to a path outside docs/, else None."""
    path, sep, fragment = target.partition("#")
    resolved = (md_file.parent / path).resolve()
    if not resolved.exists() or _DOCS_DIR in resolved.parents:
        return None
    try:
        rel = resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return None
    base = CODE_TREE_URL if resolved.is_dir() else CODE_BLOB_URL
    return f"{base}/{rel.as_posix()}{sep}{fragment}"


def _rewrite_code_links(app, docname, source):
    """Rewrite code links to CODE_BASE_URL, skipping fenced code blocks."""
    md_file = pathlib.Path(app.env.doc2path(docname))
    if md_file.suffix != ".md":
        return
    lines, fence = [], None
    for line in source[0].split("\n"):
        marker = _FENCE_RE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        elif fence is None:
            line = _MD_LINK_RE.sub(
                lambda m: _code_url(md_file, m.group(1)) or m.group(0), line
            )
        lines.append(line)
    source[0] = "\n".join(lines)


def setup(app):
    app.connect("source-read", _rewrite_code_links)
