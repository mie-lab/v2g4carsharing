"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ""
if os.path.exists("README.md"):
    with open("README.md") as fp:
        LONG_DESCRIPTION = fp.read()

scripts = []

setup(
    name="v2g4carsharing",
    version="0.0.1",
    description="vehicle to grid for car sharing",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="MIE Lab",
    author_email=("nwiedemann@ethz.ch"),
    license="GPLv3",
    url="https://github.com/mie-lab/v2g4carsharing/",
    install_requires=["numpy", "scipy", "pandas"],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages("."),
    python_requires=">=3.6",
)
