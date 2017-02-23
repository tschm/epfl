#!/usr/bin/env bash
docker run -d -p 2028:9999 -v $(pwd)/books:/jupyter tschm/ipy:v0.5