from setuptools import setup, find_packages

setup(
    name="psma_segmentator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "SimpleITK",
        "nnunet",
        "numpy",
        "argparse"
    ],
    entry_points={
        "console_scripts": [
            "psma-segmentator = psma_segmentator.cli:main"
        ]
    },
    description="PSMA PET Auto-Segmentation Tool",
    author="Jake Kendrick",
    license="MIT",
)