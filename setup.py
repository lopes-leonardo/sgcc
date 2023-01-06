# cython: language_level=3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
	name = 'lhrr',
	ext_modules=[
	Extension('lhrr',
	          sources=['sgcc/lhrr/lhrr.pyx'],
	          extra_compile_args=['--std=c++14'],
	          language='c++')
	],
	cmdclass = {'build_ext': build_ext})