from setuptools import setup, find_packages

setup(
    name="calculate_zone_time",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "supervision",
    ],
) 