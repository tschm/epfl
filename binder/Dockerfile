# Set the base image to jupyter
FROM tschm/jupyter:1.2.0 as jupyter

RUN conda install -y -c conda-forge statsmodels && conda clean -y --all

COPY --chown=jovyan:100 ./work $WORK
COPY --chown=jovyan:100 ./slides /home/jovyan/slides

WORKDIR /home/$NB_USER

#RUN mkdir /home/$NB_USER/slides && \
#RUN chown -R jovyan:100 /home/$NB_USER/slides && \
RUN chmod 775 /home/$NB_USER/slides
