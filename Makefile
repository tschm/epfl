#!make
PROJECT_VERSION := 0.3

SHELL := /bin/bash
IMAGE := tschm/epfl

.PHONY: help build jupyter tag slides clean-notebooks


.DEFAULT: help

help:
	@echo "make build"
	@echo "       Build the docker image."
	@echo "make jupyter"
	@echo "       Start the Jupyter server."
	@echo "make tag"
	@echo "       Make a tag on Github."
	@echo "make hub"
	@echo "       Push Docker Image to DockerHub."

build:
	docker-compose build jupyter

jupyter: build
	echo "http://localhost:8888"
	docker-compose up jupyter

jupyterlab: build
	echo "http://localhost:8888/lab"
	docker-compose up jupyter

tag:
	git tag -a ${PROJECT_VERSION} -m "new tag"
	git push --tags

slides:
	docker-compose up -d slides
	docker-compose exec slides jupyter nbconvert --output-dir='/slides' /home/jovyan/work/*.ipynb --to html


clean-notebooks:
	docker-compose up -d slides
	docker-compose exec slides jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace **/*.ipynb
