from setuptools import find_packages, setup
import warnings

DEPENDENCY_PACKAGE_NAMES = ["matplotlib", "torch", "tqdm", "numpy"]


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn(
            "Missing dependencies: {}. We recommend you follow "
            "the installation instructions at "
            "https://github.com/hassony2/libyana#installation".format(
                missing_dependencies
            )
        )


with open("README.md", "r") as fh:
    long_description = fh.read()

check_dependencies()

setup(
    name="libyana",
    version="0.0.1",
    author="Yana Hasson",
    author_email="yana.hasson.inria@gmail.com",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.5.0",
    description="My cross-project utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hassony2/libyana",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
