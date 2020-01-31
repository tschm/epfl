# Set the base image to jupyter
FROM lobnek/jupyterx:5.0 as jupyter

RUN conda install -y -c conda-forge matplotlib statsmodels && \
    conda clean -y --all

COPY --chown=jovyan:100 ./work /home/jovyan/work


