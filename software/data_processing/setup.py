from setuptools import setup, find_packages

setup(
    name="barbell_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=0.24.0',
        'joblib>=1.0.0',
        'openpyxl>=3.0.0'
    ]
) 