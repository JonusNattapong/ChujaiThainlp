from setuptools import setup, find_packages

setup(
    name="nlpbasiczombitx64",
    version="0.1.0",
    author="JonusNattapong",
    author_email="zombitx64@gmail.com",
    description="A basic Thai Natural Language Processing library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JonusNattapong/nlpbasiczombitx64",
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
        "nltk>=3.6.0",
        "pythainlp>=2.3.0",  # Reference for dictionaries and models
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.56.0",
    ],
    include_package_data=True,
)