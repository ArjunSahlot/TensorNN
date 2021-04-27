import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyml",
    version="0.0.1",
    author="Arjun Sahlot",
    author_email="iarjun.sahlot@gmail.com",
    description="Machine learning library made from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArjunSahlot/",
    py_modules=["pyml"],
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
)
