from setuptools import setup, find_packages

setup(
    name="stock-video-metadata",
    version="0.1.0",
    author="David Cannan",
    author_email="cdaprod@cdaprod.dev",
    description="FastAPI-based Video Metadata Enrichment API",
    url="https://github.com/Cdaprod/stock-video-metadata",
    packages=find_packages(where='app'),
    package_dir={'': 'app'},
    install_requires=[
        # Core dependencies from your requirements.txt
        "fastapi[standard]==0.115.3",
        "uvicorn[standard]==0.29.0",
        "sqlalchemy==2.0.20",
        # other core packages...
    ],
    extras_require={
        "ai": [
            "torch==2.2.2",
            "torchvision==0.17.2",
            "spacy==3.5.3",
            "transformers==4.39.3",
            # other AI-related packages...
        ],
        "dev": [
            "cython",
            "wheel",
            # other development tools...
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)