#!/bin/sh

# set -x
set -u
set -e
DIR="$( cd "$( dirname "$0")" && pwd )"
cd "${DIR}/.." || exit

# Get the operating system
os=$(uname -s)

# Get the machine architecture
arch=$(uname -m)

# Convert to lowercase and map to conda's expected names
case "$os" in
    "Linux") os="linux" ;;
    "Darwin") os="osx" ;;
    *) echo "Unsupported OS"; exit 1 ;;
esac

case "$arch" in
    "x86_64") arch="64" ;;
    "arm64") arch="arm64" ;;
    *) echo "Unsupported architecture"; exit 1 ;;
esac



while true; do
    read -p "Are you using conda? [y/n] " yn
    case $yn in
        [Yy]* ) 
            # Generate the lockfile name
            lockfile="conda-${os}-${arch}.lock"
            conda create --name myenv --file $lockfile
            break
            ;;
        [Nn]* )
            # Check if poetry is installed
            if command -v poetry >/dev/null 2>&1; then
                echo "Poetry is already installed."
                # Your logic for installing packages using poetry goes here
            else
                echo "Poetry is not installed. Installing now..."
                # Install poetry
                curl -sSL https://install.python-poetry.org | sh
            fi 
            ;;
        * ) echo "Please answer yes or no.";;
    esac
done


. ${DIR}/projects.sh
_projects=". ${PROJECTS}"
echo "Installing projects: ${_projects}"

for p in ${_projects}; do
    echo "Installing project: ${p}"
    cd "${DIR}/../${p}" || exit
    (pyenv local && poetry env use $(which python)) || poetry env use 3.11
    poetry install --sync
done