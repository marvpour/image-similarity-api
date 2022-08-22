FROM ubuntu:focal
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y curl gnupg2

# Install miniconda to /miniconda (https://gist.github.com/pangyuteng/f5b00fe63ac31a27be00c56996197597)
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

RUN echo $(ls -1 /XGEN)
# Sets the working directory in the container  
WORKDIR ./
RUN apt-get update -y

RUN conda env create -f environment.yml

COPY . ./XGEN 
CMD [ "python" , "./app.py" ]