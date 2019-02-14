# Set the base image to Ubuntu
FROM jupyter/scipy-notebook

USER root
# install cvxpy
#RUN conda install numpy && conda install -y -q -c cvxgrp cvxpy
RUN conda install -c conda-forge cvxpy=1.0.14
COPY ./jupyter_notebook_config.py /etc/jupyter/jupyter_notebook_config.py
EXPOSE 8888

USER $NB_UID
