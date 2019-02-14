# Set the base image to Ubuntu
FROM jupyter/scipy-notebook

# File Author / Maintainer
# MAINTAINER Thomas Schmelzer "thomas.schmelzer@lobnek.com"
USER root

# install cvxpy
RUN conda install -c conda-forge cvxpy=1.0.14

COPY ./books /home/$NB_USER/books

USER $NB_UID