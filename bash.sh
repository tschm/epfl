#!/usr/bin/env bash

# unfortunately this doesn't really work on Windows as pwd is not available
sudo chown -R 1000 $(pwd)/books
docker run -e NB_UID=1000 -e NB_GID=100 --user root -it -v $(pwd)/books:/home/jovyan/work -p 8888:8888 tschm/epfl /bin/bash


