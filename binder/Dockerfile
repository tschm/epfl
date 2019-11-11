# Set the base image to Ubuntu
FROM tschm/docker:release-0.6

RUN pip install --no-cache-dir cvxpy

COPY --chown=jovyan:100 ./work /home/jovyan/work
