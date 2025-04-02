from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ChujaiThainlp",
    version="1.0.0",
    author="Zombitx64",
    author_email="Zombitx64@gmail.com",
    description="Thai Natural Language Processing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonusNattapong/ChujaiThainlp",
    packages=find_packages(include=["thainlp", "thainlp.*"]),
    package_data={
        "thainlp": ["data/**/*"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "pyyaml>=6.0.1",
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
        "redis>=4.5.0",
        "prometheus-client>=0.17.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-instrumentation>=0.40b0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "tenacity>=8.2.0",
        "cryptography>=41.0.0",
        "bcrypt>=4.0.1",
        "cffi>=1.15.1",
        "pycparser>=2.21",
        "pyOpenSSL>=23.2.0",
        "pycryptodome>=3.19.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "pyasn1>=0.5.0",
        "asn1crypto>=1.5.1",
        "certifi>=2023.7.22",
        "six>=1.16.0"
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "datasets>=2.12.0",
            "evaluate>=0.4.0",
            "accelerate>=0.20.0",
            "optimum>=1.12.0"
        ],
        "audio": [
            "speechbrain",
            "pyannote.audio",
            "noisereduce"
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.9.0"
        ]
    }
)
