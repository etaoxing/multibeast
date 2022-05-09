import importlib
import os
import subprocess

from setuptools import Extension, find_packages, setup

from third_party.moolib.setup import CMakeBuild as MoolibCmakeBuild

NAME = "multibeast"
AUTHOR = f"{NAME} contributors"
URL = "https://github.com/etaoxing/multibeast"
__version__ = "0.0.1"

cwd = os.path.dirname(os.path.abspath(__file__))
packages = find_packages(exclude=["tests*", "examples*"]) + ["multibeast.examples"]
package_dir = {"": ".", "multibeast.examples": "examples"}
ext_modules = []
cmdclass = {}

install_requires = [
    "torch>=1.9.1",  # make sure this is the first entry
    "hydra-core>=1.0.0",
    "hydra-colorlog>=1.0.0",
    "hydra-submitit-launcher>=1.1.1",
    "wandb>=0.10.31",
    "pyyaml",
]

moolib_spec = importlib.util.find_spec("moolib")
found_moolib = moolib_spec is not None
if os.getenv("SKIP_MOOLIB_BUILD"):
    install_requires += [  # pypi version
        "moolib",
    ]
else:
    install_requires += [
        "cmake>=3.14.4",  # for building moolib
    ]
    if os.getenv("RELEASE_BUILD"):
        # bundle a build of moolib
        packages += ["moolib", "moolib.examples.common"]
        package_dir.update({"moolib": "third_party/moolib/py/moolib", "moolib.examples": "third_party/moolib/examples"})
        ext_modules += [Extension("moolib._C", sources=[])]
        cmdclass["build_ext"] = MoolibCmakeBuild

        # enforce a strict pytorch version
        install_requires[0] = "==".join(install_requires[0].split(">="))
    else:
        # support pip installing in editable mode
        # otherwise top-level moolib module will not be found
        install_requires += [
            f"moolib @ file://{cwd}/third_party/moolib#egg=moolib",
        ]

install_requires += [
    "tinyspace@git+http://github.com/etaoxing/tinyspace#egg=tinyspace",
    "tabulate",
    "coolname",
]

extras_deps = {
    "tests": [
        "pre-commit>=2.0.1",
        # Reformat
        "black>=19.10b0",
        # Lint code
        "flake8>=3.7",
        # Find likely bugs
        "flake8-bugbear>=20.1",
        # Docstrings style,
        "flake8-docstrings>=1.6.0",
        # Run tests and coverage
        "pytest>=5.3",
        "pytest-benchmark>=3.1.0",
        "pytest-order>=1.0.1",
        "pytest-cov",
        "pytest-xdist",
        # Type check
        "pyright",
        # Sort imports
        "isort>=5.0",
    ],
    "docs": [
        "sphinx==4.4.0",
        "sphinx-autobuild",
        "myst-parser",
        # # Jupyter notebooks
        # "nbsphinx",
        # For spelling
        "sphinxcontrib-spelling",
        # Type hints support
        "sphinx-autodoc-typehints",
        # Extras
        "sphinx-design",
        "sphinx-copybutton",
        "sphinx-inline-tabs",
        "sphinxcontrib-trio",
        "sphinxext-opengraph",
        # Theme
        "furo",
    ],
}

extras_deps["all"] = [item for group in extras_deps.values() for item in group]


if __name__ == "__main__":
    with open("README.md") as f:
        long_description = f.read()
    sha = "unknown"
    version = __version__

    if os.getenv("RELEASE_BUILD") or (os.getenv("READTHEDOCS") and os.getenv("READTHEDOCS_VERSION_TYPE") == "tag"):
        sha = version

        if not os.getenv("SKIP_MOOLIB_BUILD"):
            # platform-specific
            cuda = None
            try:
                nvcc_ver = os.popen('nvcc --version | egrep -o "V[0-9]+.[0-9]+.[0-9]+" | cut -c2-').read().strip()
                cu_major, cu_minor, cu_patch = nvcc_ver.split(".")
                cuda = f"{cu_major}{cu_minor}"
            except Exception:
                pass

            if cuda is not None:
                version = f"{__version__}+cu{cuda}"
    else:
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
        except subprocess.CalledProcessError:
            pass
        version += ".dev0+" + sha[:7]

    version_path = os.path.join(cwd, NAME, "version.py")
    with open(version_path, "w") as f:
        f.write(f'__version__ = "{version}"\n')
        f.write(f'commit = "{sha}"\n')

    print(f"Building {NAME}-{version}")
    print(packages)

    setup(
        name=NAME,
        version=version,
        description="",
        author="etaoxing",
        url=URL,
        download_url=f"{URL}/tags",
        license="MIT",
        packages=packages,
        package_dir=package_dir,
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
        install_requires=install_requires,
        extras_require=extras_deps,
        python_requires=">=3.7",
        zip_safe=False,
    )
