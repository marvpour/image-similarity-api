# Image Similarity

Project for finding 10 most similar images

# Notes

 The first time it is running, takes around 2h to dowanload all the images.
 The POST request takes a url as the body and will return 10 links that are closest to the input image link.


### Running

You can install Anaconda or Miniconda.

[install guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Initial package installation

You must create a virtual environment from the environment.yml file included in the project.
To do this either use pycharm, or run ```conda env create -f environment.yml```.

### Installing additional packages

To install new packages you must add them to the environment.yml file with the version number. Then run
```conda env update --file environment.yml``` in your activated environment (see [Running](#running) for how to activate
your environment).

### Exporting your environment

You can export your environment by running ```conda env export --from-history > environment.yml```. This should not be
done unless something is wrong with the environment.yml and you need to recreate it. Note that some packages may need
versions added and that some channels may be missing.

## Running

Activate your Conda environment by running ```conda activate image-similarity```. Then run ```python app.py```
to start the server.


## Docker

Note: make sure to change account and version numbers as needed

### Build (local)

docker build -t home-assessment:1.0.0 .
