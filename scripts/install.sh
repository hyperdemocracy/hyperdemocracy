#!/bin/sh

# set -x
set -u
set -e
DIR="$( cd "$( dirname "$0")" && pwd )"
cd "${DIR}/.." || exit

. ${DIR}/projects.sh
_projects=". ${PROJECTS}"
echo "Installing projects: ${_projects}"

for p in ${_projects}; do
    echo "Installing project: ${p}"
    cd "${DIR}/../${p}" || exit
    (pyenv local && poetry env use $(which python)) || poetry env use 3.11
    poetry install --sync
done