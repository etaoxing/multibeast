# Contributing

## Making a pull request

Fork the repo on GitHub.

```bash
# clone fork
git clone https://github.com/USER/multibeast.git

# add original repo
git remote add upstream https://github.com/etaoxing/multibeast.git

# keep fork up to date
git checkout master
git pull --rebase upstream master
git push
```

Make code changes in a branch.
```bash
# create feature branch
git checkout -b BRANCH
git push -u origin BRANCH

# setup dev environment
conda create -n PACKAGE python=3.7
conda activate PACKAGE
pip install -e ".[all]"
```

Run linting, style formatting, and tests.
```bash
flake8 .
isort .
black .
pytest tests/ -v -rP --forked --cov
```

Document your code, following a [looser version](https://drake.mit.edu/styleguide/pyguide.html) of Google style docstrings.

Open a new pull request on GitHub, and add appropriate labels.

## Building documentation

```bash
cd docs/
make clean
make html # or `make livehtml` for autobuild
```

## Releasing

> Make sure PYPI_TOKEN secret is added. Go to "Settings" > "Secrets" > "Actions", and then click "New repository secret".

```bash
# regenerates `version.py`
RELEASE_BUILD=1 python setup.py check sdist bdist_wheel

export TAG=$(python -c 'from multibeast.version import __version__; print(__version__)')
git add setup.py multibeast/version.py
git commit -m "Release ${TAG}" && git push

# make sure that tests pass, then proceed

git tag "${TAG}" -m "${TAG}"
git push --tags
```

After tagging a release, bump `__version__` in [`setup.py`](setup.py).
```bash
# regenerates `version.py`
RELEASE_BUILD=1 pip install -e .  # set RELEASE_BUILD so not including commit hash
export TAG=$(python -c 'from multibeast.version import __version__; print( __version__)')
git add setup.py multibeast/version.py
git commit -m "Bump to ${TAG} [skip ci]" && git push
```

Then go to the repo's GitHub releases [page](https://github.com/etaoxing/multibeast/releases), and publish the drafted release notes.

If you need to fix a failed release, delete the tag and the corresponding release (if published) from GitHub.
Also, delete the tag from your local clone of the repo with `git tag -l | xargs git tag -d && git fetch -t`.

### Publishing on readthedocs

Go to the [readthedocs dashboard](https://readthedocs.org/dashboard/import/?) and import the repo.

Then go to "Admin" -> "Automation Rules" -> "Add Rule", and enter the following:

- **Description:** Publish new versions from tags
- **Match:** SemVer versions
- **Version:** Tag
- **Action:** Activate version

and click "Save".

Go to "Admin" -> "Advanced Settings" -> "Default version", change to "stable", and click "Save".

Then go to "Versions" -> "Latest", and check "hidden".
