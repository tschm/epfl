# Set the base image to Ubuntu
FROM jupyter/scipy-notebook

# File Author / Maintainer
MAINTAINER Thomas Schmelzer "thomas.schmelzer@lobnek.com"

# install cvxpy
RUN conda install numpy && conda install -y -q -c cvxgrp cvxpy
