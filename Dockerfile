# We use the techniques described here:
# (https://pythonspeed.com/articles/conda-docker-image-size/)
# to minimize the impact of conda on our build. The goal here is to minimize image size
# and re-implementation of build logic from habitat-sim and conda
FROM  continuumio/miniconda3 AS build

# Install the package as normal:
RUN conda create -n l2m

# Install conda-pack (per https://pythonspeed.com/articles/conda-docker-image-size/)
# and habitat-sim
RUN conda install \
  conda-pack \
  habitat-sim==0.1.7 \
  # for running habitat-sim headless:
  headless==1.0=0 \
  # required by habitat-sim and habitat-lab:
  pytorch==1.8.0 \
  # for torch:
  cudatoolkit==11.1.1 \
  -c conda-forge -c pytorch -c aihabitat

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n l2m -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM nvidia/cudagl:11.2.0-devel-ubuntu20.04

# Copy /venv from the previous stage:
COPY --from=build /venv /venv
COPY --from=build /opt/conda/ /opt/conda/

# add /venv to Path for access to python and pip
ENV PATH="/venv/bin:/opt/conda/bin/:$PATH"

WORKDIR "/habitat-lab"
COPY . .
