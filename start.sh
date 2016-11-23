#!/usr/bin/env bash

port=$1
host=":9999"

docker-compose run -d -p $port$host pytalk
