# Set the base image to jupyter
FROM tschm/jupyter:1.2.0 as jupyter

RUN conda install -y -c conda-forge statsmodels && conda clean -y --all

COPY --chown=jovyan:100 ./work $WORK

WORKDIR /home/$NB_USER

#RUN mkdir /home/$NB_USER/slides && \
#RUN chown -R jovyan:100 /home/$NB_USER/slides && \
#RUN chmod 775 /home/$NB_USER/slides


#### Here the test-configuration
FROM jupyter as slides

RUN mkdir /home/$NB_USER/slides && \
    chown -R jovyan:100 /home/$NB_USER/slides

CMD jupyter nbconvert --output-dir=home/$NB_USER/slides /home/$NB_USER/work/*.ipynb --to html

