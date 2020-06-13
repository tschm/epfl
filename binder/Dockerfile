# Set the base image to jupyter
FROM tschm/jupyter:1.2.0 as jupyter

#RUN conda install -y -c conda-forge statsmodels && conda clean -y --all

COPY --chown=jovyan:100 ./work $WORK

WORKDIR /home/$NB_USER

