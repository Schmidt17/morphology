from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
	Extension("MC", sources = ["MC.pyx"], include_dirs = [numpy.get_include()])
]

setup (
	name = "MC",
	ext_modules = cythonize(ext_modules)
)