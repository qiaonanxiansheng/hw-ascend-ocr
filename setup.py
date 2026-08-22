"""Setup script for ascend-ocr."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ascend-ocr",
    version="3.0.0",
    author="ascend-ocr contributors",
    description="Native, high-cohesion OCR engine for Ascend NPU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ascend-ocr",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples", "ascend"]),
    package_data={
        "ascend_ocr": ["configs/*"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python-headless>=4.5.0",
        "pyclipper>=1.2.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "python-multipart>=0.0.6",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
)
