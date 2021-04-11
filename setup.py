import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rl_flows",
    version="0.0.1",
    author="",
    author_email="levmckinney@gmail.com",
    description="Research on cellular atomata for image inpainting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'torch',
        'ray[rllib]',
        'GPy',
        'sklearn',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)