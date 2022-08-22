FROM continuumio/miniconda3:latest

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y curl gnupg2 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/miniconda/bin:${PATH}

# Sets the working directory in the container  
WORKDIR ./XGEN
COPY environment.yml .
RUN conda env create -f environment.yml

COPY app.py .
COPY src .
COPY in .

CMD [ "python" , "./app.py" ]