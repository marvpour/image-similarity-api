# onboarding-server

Project for finding 10 most similar images

## Conda

### Install Conda

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

Activate your Conda environment by running ```conda activate onboarding-server```. Then run ```python app.py```
to start the server.


## Docker

Note: make sure to change account and version numbers as needed

### Build and Push (local)

docker build -t localintelcr.azurecr.io/onboarding-server:1.0.0 .

docker login localintelcr.azurecr.io

docker push localintelcr.azurecr.io/onboarding-server:1.0.0

### Pull and Start (VM)

sudo docker login localintelcr.azurecr.io

sudo docker pull localintelcr.azurecr.io/onboarding-server:1.0.0

sudo docker stop onboarding-server

sudo docker rm onboarding-server

sudo docker run -d -p 5562:5562 --restart=always --name=onboarding-server localintelcr.azurecr.io/onboarding-server:1.0.0

Note: you may want to mount the logs, instance, and tmp directories onto your local machine (use the -v, e.g. -v /home/localintel/onboarding-server/instance:/onboarding-server/instance)
