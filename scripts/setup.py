import os
from os.path import join as path_join
import setuptools

PARDIR = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
TNN = path_join(PARDIR, "tensornn")

with open(path_join(PARDIR, "README.md")) as f:
    long_description = f.read()

with open(path_join(PARDIR, "requirements.txt")) as f:
    reqs = f.read().strip().split("\n")


setuptools.setup(
    name="tensornn",
    version=print(os.getenv("PYPI_VERSION")),
    author="Arjun Sahlot",
    author_email="iarjun.sahlot@gmail.com",
    description="Machine learning library made from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GNU GPL v3",
    url="https://github.com/ArjunSahlot/tensornn",
    keywords=["Machine Learning"],
    py_modules=["tensornn"],
    packages=setuptools.find_packages(),
    install_requires=reqs,
    python_requires=">=3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
