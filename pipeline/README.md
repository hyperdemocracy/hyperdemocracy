# Pipeline

1. acquisition -> fetches raw data and adds to s3 + metadata store
2. parse XML to pydantic
3. langchain document loading / splitting
4. embedding -> write to HF, lance

## Overview

starting with no data:

option 1 - use the remote dataset in S3
option 2 - create a local dataset

## Run data acquisition

we have a repo of datasets that have their own container images
deploy the container image to some compute (e.g. hyperdemocracy/congress:latest to aws lamba or ec2)
pipeline orchestration happens in this package `hyperdemocracy/pipeline`
pick up a configuration, for example:
  - raw data -> s3://hyperdemocracy-dev/data