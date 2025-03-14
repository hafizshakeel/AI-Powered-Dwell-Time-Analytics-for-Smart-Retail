from setuptools import find_packages, setup

setup(
    name='retail_dwell_time',
    version='0.1.0',
    author='Hafiz Shakeel Ahmad Awan',
    author_email='hafizshakeel1997@gmail.com',
    description='A comprehensive solution for tracking and analyzing how long people spend in defined zones',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
