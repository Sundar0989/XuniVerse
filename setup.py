import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="xverse", # Replace with your own username
    version="1.0.3",
    author="Sundar Krishnan",
    author_email="sundarstyles89@gmail.com",
    description="xverse short for X uniVerse is collection of transformers for feature engineering and feature selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sundar0989/XuniVerse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5.*',
    license='MIT',
    install_requires=[
      'numpy>=1.11.3',
      'scikit-learn>=0.19.0',
      'scipy>=0.19.0',
      'statsmodels>=0.6.1',
      'pandas>=0.21.1'
    ]
)
