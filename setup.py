from setuptools import setup, find_packages
import sys

req_version = (3, 6)

long_description = """
Autoclf is a machine learning toy-concept to show
how to automate  tabular data classification with calibrated estimators,
while exploring the scikit-learn and Keras ecosystems.
"""

if sys.version_info < req_version:
    sys.exit("Python 3.6 or higher required to run this code, " +
             sys.version.split()[0] + " detected, exiting.")

setup(
    name="autoclf",
    version="0.1.7",
    author="Simon Carozza",
    author_email="simoncarozza@gmail.com",
    description="A tabular data classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SimonCarozza/autoclf",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Other OS"
    ],
    install_requires=[
        "scikit-learn>=0.20.1",
        "matplotlib>=3.0.2",
        "pandas",
        "keras",
        "tensorflow"
    ],
    extras_require={
        "xgboost": ["py-xgboost==0.8"],
        "Hyperparam Opt.": ["scikit-optimize"]},
    package_data={
        "autoclf": [
            "datasets/*.csv", "datasets/ReadMe.txt", 
            "datasets/*.zip"]})
