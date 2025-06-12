"""DeepRetail setup"""

from os import path
import setuptools

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    author="Yves R. Sagaert",
    author_email="yves.r.sagaert@gmail.com",
    name="DeepRetail",
    license="GNU GPLv3",
    description='Forecasting package for retail using Deep Learning AI.',
    version='v0.1.0',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yForecasting/DeepRetail",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    # Core dependencies with compatible versions to avoid numpy binary incompatibility
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.0.3",
        "scipy==1.11.4",
        "scikit-learn==1.2.1",
        # "dask[dataframe]>=2023.2.0",
        # "distributed>=2023.2.0",
        "matplotlib==3.2.2",
        "statsforecast==1.4.0",
        "statsmodels==0.13.5",
        "sktime==0.24.1",
        "numba>=0.56.0",
        "openpyxl>=3.0.0",
        "tsfeatures>=0.4.0",
        "python-dateutil>=2.8.0",
        "tqdm>=4.64.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],
)
