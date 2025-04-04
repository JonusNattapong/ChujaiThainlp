"""
Setup configuration for ChujaiThainlp package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chujaithai-nlp",
    version="2.0.0",
    author="ThaiNLP",
    author_email="contact@thainlp.org",
    description="Advanced Thai Natural Language Processing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thainlp/ChujaiThainlp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Thai",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.21.0",
        "sentence-transformers>=2.2.0",
        "pandas>=1.3.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
        "rouge-score>=0.1.0",
        "sacrebleu>=2.0.0",
        "psutil>=5.8.0",  # For resource monitoring
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-markdown-tables>=0.0.15",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",  # CUDA 11.1
        ],
    },
    package_data={
        "thainlp": [
            "tokenization/data/*.txt",
            "tokenization/data/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "thainlp=thainlp.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/thainlp/ChujaiThainlp/issues",
        "Documentation": "https://chujaithai-nlp.readthedocs.io/",
        "Source": "https://github.com/thainlp/ChujaiThainlp",
    },
)
