from setuptools import setup, find_packages

setup(
    name="CellDeathSpreading",
    version="0.1.0",
    description="Analysis and visualization tools for cell death spreading experiments",
    author="Esraan",
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.1",
        "matplotlib==3.8.3",
        "scipy==1.12.0",
        "scikit-image==0.22.0",
        "networkx==3.1",
        "imageio==2.33.1",
        "tifffile==2023.4.12",
        "pillow==10.2.0",
        "ipython==8.22.2",
        "ipykernel==6.29.3",
        "jupyter-client==8.6.1",
        "jupyter-core==5.7.2",
        "pyzmq==25.1.2",
        "tornado==6.3.3",
        "python-dateutil==2.8.2",
        "pytz==2023.3.post1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "notebooks": [
            "jupyter",
        ],
    },
    include_package_data=True,
)