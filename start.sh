#!/usr/bin/env bash
sudo chown -R 1000 books

docker run -e NB_UID=1000 -e NB_GID=100 --user root -v $(pwd)/books:/home/jovyan/work -p 8888:8888 tschm/epfl start-notebook.sh --NotebookApp.token=''



