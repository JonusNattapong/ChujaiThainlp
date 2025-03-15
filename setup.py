from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chujaithainlp",
    version="0.4.0",
    author="Zombitx64",
    author_email="zombitx64@gmail.com",
    description="Thai Natural Language Processing Library with Advanced Deep Learning Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zombitx64/ChujaiThainlp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: Thai",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "pyyaml>=6.0.1",
        "scikit-learn>=1.0.0",
        "pythainlp>=3.1.0",
    ],
    extras_require={
        "full": [
            "pythainlp>=3.1.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "datasets>=2.12.0",
            "evaluate>=0.4.0",
            "accelerate>=0.20.0",
            "optimum>=1.12.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.0.0",
            "nltk>=3.8.0",
        ],
        "transformers": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
        ],
        "data": [
            "datasets>=2.12.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
