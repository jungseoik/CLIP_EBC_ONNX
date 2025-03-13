from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clipebc",
    version="0.1.0",
    author="jungseoik",
    author_email="si.jung@pia.space",
    description="Crowd counting with CLIP-EBC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jungseoik/CLIP_EBC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.12.4",
    install_requires=[
        "einops==0.7.0",
        "ftfy==6.1.3",
        "numpy==1.26.4",
        "Pillow==10.2.0",
        "regex==2023.12.25",
        "scipy==1.12.0",
        "tensorboardX==2.6.2.2",
        "timm==0.9.16",
        "torch==2.2.1",
        "torchvision==0.17.1",
        "tqdm==4.66.2",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
    extras_require={
        "app": ["streamlit", "gradio"],
        "dev": ["pytest", "black", "isort"],
    }
)