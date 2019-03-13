# Set the base image to Ubuntu
FROM lobnek/jupyter:v2.3

USER root
RUN conda install -y -c conda-forge matplotlib && conda clean -y --all
USER $NB_USER


COPY ./work ${HOME}/work
