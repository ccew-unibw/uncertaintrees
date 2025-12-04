# We start from a very simple Rocker image as this should come with
# R pretty much preconfigured. We fix the version to 4.3.2 as we've
# used R 4.3.2 for the distributional random forest package
FROM rocker/r-ver:4.3.2
WORKDIR /usr/src/app
# Install the drf package; this will perform a binary by default with the new rocker/r-ver iamges
# TODO: consider fixing the version number to 1.1.0, but unfortunately not trivial in R :/
RUN R -q -e 'install.packages("drf")' 
# Install python and all required system dependencies (the latter should probably be shortened to some major parent libs)
RUN apt-get update && apt-get install -y software-properties-common gcc curl git wget \
    libbz2-dev liblzma-dev libpcre2-dev zlib1g-dev libicu-dev  && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-venv build-essential
# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
# This is required to solve a depency issue with cffi and rpy2 in older pip versions
RUN pip install --upgrade pip
# Copy the requirements file into the build process
COPY requirements.tx[t] ./
# Install all required python packages
RUN pip install --no-cache-dir -r requirements.txt
# Define an entrypoint
ENTRYPOINT ["/bin/bash", "-l", "-c"]