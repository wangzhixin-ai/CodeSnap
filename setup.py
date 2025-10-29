"""Setup script for codesnap package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="codesnap",
    version="0.1.0",
    author="Wang Zhixin",
    author_email="wangzx@sii.edu.cn",
    description="A comprehensive debugging tool for ML/DL with complete reproducibility tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangzhixin-ai/CodeSnap",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Debuggers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        "torch": ["torch"],
        "numpy": ["numpy"],
        "all": ["torch", "numpy"],
        "dev": ["pytest", "pytest-cov"],
    },
    keywords="debugging tensor numpy pytorch machine-learning deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/wangzhixin-ai/CodeSnap/issues",
        "Source": "https://github.com/wangzhixin-ai/CodeSnap",
    },
)
