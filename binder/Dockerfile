# Set the base image to Ubuntu
FROM jupyter/scipy-notebook:83ed2c63671f

USER root

# install cvxpy
RUN conda install -c conda-forge cvxpy=1.0.14

COPY ./books ${HOME}/books

USER ${NB_USER}