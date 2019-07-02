# Set the base image to Ubuntu
FROM lobnek/jupyter:v2.8

RUN pip install --no-cache-dir matplotlib statsmodels

COPY --chown=beakerx:beakerx ./work ${WORK}

WORKDIR ${WORK}
