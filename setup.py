from setuptools import find_packages, setup

DESCRIPTION = "Toolkit for machine learning development"

# Required packages
# full tensorflow is too big for readthedocs's builder
REQUIRES = [
    'seaborn',
    'matplotlib',
    'pandas',
    'numpy',
    # "numpy>=1.14",
    # "scipy>=1.1.0",
    # "networkx>=2.2",
    # "scikit_learn>=0.20",
    # "matplotlib>=2.2",
    # "gensim>=3.4.0",
    # "pandas>=0.24",
]


setup(
    name='devkit',
    version='0.1.0',
    description=DESCRIPTION,
    author='jasper gui',
    author_email="glc_luck@outlook.com",
    license='MIT',
    # python_requires=">=3.5.0, <3.9.0",
    install_requires=REQUIRES,
    packages = find_packages(),
    include_package_data=True,
    package_data = {'': ['*.zip']},
    zip_safe = False,
    #--------------------------------------------------------------------------
    #  MODIFY: Remove this and delete the file if not required.
    #  Rename otherwise!
    #--------------------------------------------------------------------------
    #scripts = [
    #    'bin/etlctl',  # the etl pipeline controller cli
    #    'bin/modelctl',  # the model pipeline controller cli
    #],
)
