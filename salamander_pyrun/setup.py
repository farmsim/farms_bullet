#!/usr/bin/env python
""" Setup script """

from setuptools import setup


setup(
    name="salamander_pyrun",
    version="0.1",
    author="Jonathan Arreguit",
    author_email="jonathan.arreguitoneill@epfl.ch",
    description="Salamander pyrun",
    # license="BSD",
    keywords="salamander model pyrun gazebo",
    # url="",
    packages=['salamander_pyrun'],
    # long_description=read('README'),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
    scripts=['scripts/salamander_run.py'],
    # package_data={'salamander_pyrun': []},
    include_package_data=True
)
