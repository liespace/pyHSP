import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='heurisp',
    version='0.2.0',
    packages=setuptools.find_packages(),
    description='Heuristics for Sampling-based Path Planner',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gabel Liemann',
    author_email='troubleli233@gmail.com',
    url='https://github.com/liespace/pyHSP',
    license='MIT',
    install_requires=[
        'numpy',
        'numba',
        'reeds_shepp',
        'matplotlib',
        'opencv-python',
        'scipy'],
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
)
