from setuptools import setup, find_packages

setup(
    name="psma_segmentator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
            'torch>=2.1.2',
            'numpy<2',
            'SimpleITK',
            'nibabel>=2.3.0',
            'tqdm>=4.45.0',
            'p_tqdm',
            'xvfbwrapper',
            'nnunetv2>=2.2.1',
            'requests==2.27.1;python_version<"3.10"',
            'requests;python_version>="3.10"',
            'rt_utils',
            'dicom2nifti',
            'pyarrow',
            'pydicom'
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