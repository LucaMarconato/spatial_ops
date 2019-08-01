#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# spatial_ops documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import shutil
import sys
import subprocess
sys.path.insert(0, os.path.abspath('..'))

import spatial_ops
import sphinx_gallery
import sphinx_rtd_theme



def erase_folder_content(folder):
    if os.path.isdir(folder):
        for file_object in os.listdir(folder):
            file_object_path = os.path.join(folder, file_object)
            if os.path.isfile(file_object_path):
                os.unlink(file_object_path)
            else:
                shutil.rmtree(file_object_path) 


on_rtd  = os.environ.get('READTHEDOCS', None) == 'True'
on_travis = os.environ.get('TRAVIS', None) == 'True'
on_ci = on_rtd or on_travis


package_name = "spatial_ops"
this_dir = os.path.dirname(__file__)
py_mod_path  = os.path.join(this_dir, '../')
package_dir = os.path.join(py_mod_path, package_name)
template_dir =  os.path.join(this_dir, '_template')
if True:
    apidoc_out_folder =  os.path.join(this_dir, 'api')
    erase_folder_content(apidoc_out_folder)
    arglist = ['sphinx-apidoc','-o',apidoc_out_folder,package_dir,'-P']
    subprocess.call(arglist, shell=False)




# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.napoleon'
]



sphinx_gallery_conf = {
    'doc_module': ('spatial_ops',),
    # path to your examples scripts
    'examples_dirs': '../examples',
    # path where to save gallery generated examples
    'gallery_dirs': 'auto_examples',
    'backreferences_dir': 'api',
    'reference_url': {'spatial_ops': None},
}
html_theme = "sphinx_rtd_theme"
#html_theme = "classic"
html_theme_path = [
    sphinx_rtd_theme.get_html_theme_path(),
    'mytheme',
    '.'
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'spatial_ops'
copyright = u"2019, Thorsten Beier"
author = u"Thorsten Beier"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = spatial_ops.__version__
# The full version, including alpha/beta/rc tags.
release = spatial_ops.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False




# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'spatial_opsdoc'


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'spatial_ops.tex',
     u'spatial_ops Documentation',
     u'Thorsten Beier', 'manual'),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'spatial_ops',
     u'spatial_ops Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'spatial_ops',
     u'spatial_ops Documentation',
     author,
     'spatial_ops',
     'One line description of project.',
     'Miscellaneous'),
]



