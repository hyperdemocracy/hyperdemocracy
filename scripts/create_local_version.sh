#!/bin/sh

DIR="$( cd "$( dirname "$0" )" && pwd )"
cd "${DIR}/.." || exit

# first run directly, to have script stop if dunamai isn't available (for example if not installed, or running in wrong virtual env)
dunamai from any
VERSION=$(dunamai from any)
echo $VERSION

# all python packages, in topological order
. ${DIR}/projects.sh
_projects=". ${PROJECTS}"
echo "Running on following projects: ${_projects}"
if [ "$(uname)" = "Darwin" ]; then export SEP=" "; else SEP=""; fi
for p in $_projects
do
  echo "Creating local version of ${p}"
  echo "$VERSION" > "${p}/VERSION"
  sed -i$SEP'' "s/^version = .*/version = \"$VERSION\"/" "$p/pyproject.toml"
done
sed -i$SEP'' "s/^__version__.*/__version__ = \"$VERSION\"/" hyperdemocracy/hyperdemocracy/__init__.py
sed -i$SEP'' "s/^__version__.*/__version__ = \"$VERSION\"/" legisqa/legisqa/__init__.py
