from setuptools import setup, find_packages
from codecs import open
from os import path

import rhymediscovery


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rhymediscovery',
    version=rhymediscovery.__version__,
    description='Unsupervised Discovery of Rhyme Schemes',
    long_description=long_description,
    url='https://github.com/jvamvas/rhymediscovery',
    author='Jannis Vamvas',
    author_email='jannis.vamvas@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=['rhymediscovery'],
    install_requires=['numpy'],
    package_data={
        'rhymediscovery': ['data/*'],
    },
    data_files=[('tests', ['tests/data/sample.txt', 'tests/data/sample.pgold'])],
    entry_points={
        'console_scripts': [
            'find_schemes=rhymediscovery.find_schemes:main',
            'evaluate_schemes=rhymediscovery.evaluate_schemes:main',
        ],
    },
)
