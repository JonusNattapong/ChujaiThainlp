from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thainlp",
    version="0.2.1",
    author="Zombitx64",
    author_email="zombitx64@gmail.com",
    description="Thai Natural Language Processing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zombitx64/thainlp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: Thai",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "full": [
            "pythainlp>=3.1.0",
            "torch>=1.7.0",
            "transformers>=4.0.0",
        ],
        "pythainlp": [
            "pythainlp>=3.1.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
        ],
    },
)