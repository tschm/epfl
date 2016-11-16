# Set the base image to Ubuntu
FROM lobnek/ipy:v0.0.3 

# File Author / Maintainer
MAINTAINER Thomas Schmelzer "thomas.schmelzer@gmail.com"

# install additional packages not provided by the base image...
RUN conda install -y statsmodels scikit-learn tensorflow && conda install -y -q -c cvxgrp cvxpy
