# Usage of environments

For the environment for record 1-11 `speedrunner-1-11` and record 12-18 `speedrunner-12-18`, they can be installed simply via `conda` and `pip`, run the following commands.

```bash
# record 1-11
cd speedrunner-1-11
conda env create -f environment-1-11.yml
conda activate environment-1-11
pip install -r pip_requirements-1-11.txt


# record 12-18
cd speedrunner-12-18
conda env create -f environment-12-18.yml
conda activate environment-12-18
pip install -r pip_requirements-12-18.txt
```


For the environment for record 19-21 `speedrunner-19-21`, as we used a nightly build of `torch`, we will need to install it from conda-pack, run the following commands.

```bash
# record 19-21 

## conda-unpack to rebuild the environment
mkdir -p ~/path/to/envs/environment-19-21
tar xzvf speedrunner-19-21.tar.gz -C ~/path/to/envs/environment-19-21
~/path/to/envs/environment-19-21/bin/conda-unpack

## activate the environment
source ~/path/to/envs/environment-19-21/bin/activate

## alternatively, the following command can be used to link the environment to existing conda, then conda activate can be used
conda config --append envs_dirs ~/path/to/envs
conda activate environment-19-21
```