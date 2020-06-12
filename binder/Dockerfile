# Set the base image to jupyter
FROM tschm/jupyter:1.2.0 as jupyter

RUN conda install -y -c conda-forge statsmodels && conda clean -y --all

COPY --chown=jovyan:100 ./work $WORK

RUN mkdir /home/$NB_USER/slides && \
    #fix-permissions /home/$NB_USER
    chown -R jovyan:100 /home/$NB_USER/slides && \
    chmod 755 /home/$NB_USER/slides
