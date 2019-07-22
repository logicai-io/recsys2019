from Cython.Build import cythonize
from numpy import get_include
from setuptools import setup, Extension

extensions = [Extension(name="recsys.mrr", sources=["recsys/*.pyx"])]

setup(
    name="recsys",
    version="0.1",
    author="LogicAI",
    packages=["recsys"],
    ext_modules=cythonize(extensions, include_path=[get_include()]),
    include_dirs=[get_include()],
)
