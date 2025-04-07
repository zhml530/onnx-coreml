from setuptools import setup, find_packages
from os import path
import sys
from onnx_coreml import __version__

VERSION = __version__

here = path.abspath(path.dirname(__file__))

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    with open('README.md', "rb") as f:
        long_description = f.read().decode("UTF-8")


if sys.version_info[0] == 2:
    # Mypy doesn't work with Python 2
    mypy = []
elif sys.version_info[0] == 3:
    mypy = ['mypy==0.560']


setup(
    name='onnx-coreml',
    version=VERSION,
    packages=find_packages(exclude=['contrib', 'docs', 'test', 'example']),
    description='Convert ONNX (Open Neural Network Exchange)'
                'models into Apple CoreML format.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/onnx/onnx-coreml/',
    author='ONNX-CoreML Team',
    author_email='onnx-coreml@apple.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    keywords='onnx coreml machinelearning ml coremltools converter neural',
    install_requires=[
        'click',
        'numpy',
        'sympy',
        'onnx==1.12.0',
        'typing>=3.6.4',
        'typing-extensions>=3.6.2.1',
        'coremltools==4.0',
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'pytest-cov',
        'Pillow'
    ],
    extras_require={
        'mypy': mypy,
    },
    entry_points={
        'console_scripts': [
            'convert-sld = onnx_coreml.bin.convert_sld:onnx_to_coreml',
            'convert-landmark = onnx_coreml.bin.convert_landmark:onnx_to_coreml',
        ]
    },
)
