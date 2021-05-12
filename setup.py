#!/usr/bin/env python

from setuptools import find_packages, setup


setup(name='delayed_fb_cb',
      version='0.1',
      description='Delayed Feedback for Contextual Bandits',
      author='Andrei Kapustin and Jesse Swanson and William Egan',
      author_email='js11133@nyu.edu',
      package_dir={"": "src"},
      packages=find_packages("src"),
      install_requires=[
          "numpy == 1.20.3"
      ],
      )
