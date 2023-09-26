# hyperdemocracy

Hyperdemocracy

Tool for creating LLM applications using multiple different government datasets.

## Features

- Maintained public datasets of lots of awesome government sources
- Data pipeline for loading, integrating, and running inference on government datasets
- Web UI for exploring and visualizing data
- Python client library for integrating hyperdemocracy in your own applications


## Pre-requisites
- Python >= 3.11
- poetry >= 1.0.0

# Monorepo 
We're using a monorepo so we can develop multiple apps for different environments using a shared set of dependencies and tools.
In our case, `hyperdemocracy` is the core package that will be distributed on pypi.
Inspired by this [example](https://gerben-oostra.medium.com/python-poetry-mono-repo-without-limitations-dd63b47dc6b8)

# Setup

- Install dependencies with `./scripts/install.sh`, select whether or not you are using conda
- To run a script `poetry run python yourscript.py`
- To add a new package, from the root level of the repo, `poetry new yourpackagename`



# Versioning
Commitizen is in place to enforce semantic versioning. All monorepo components that depend on each other should be versioned together, i.e. if UI depends on hyperdemocracy, and hyperdemocracy gets bumped to `1.2.3`, UI should also be versioned `1.2.3`