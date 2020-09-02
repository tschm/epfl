# Set the base image to jupyter
FROM tschm/jupyter:1.3.1 as jupyter

#RUN conda install -y -c conda-forge statsmodels && conda clean -y --all

COPY --chown=jovyan:100 ./work $WORK

WORKDIR /home/$NB_USER

