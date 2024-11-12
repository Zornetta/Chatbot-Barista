# setup.py
from setuptools import setup, find_packages

setup(
    name="starbucks_chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'spacy>=3.0.0',
        'scikit-learn>=1.0.0',
        'numpy>=1.20.0',
        'joblib>=1.0.0'
    ],
    python_requires='>=3.7',
)