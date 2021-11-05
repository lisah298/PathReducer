Machine Learning of Hydrogen Combustion Reaction. 

## Installation and Dependencies
The pypi or conda installation is not available for this library yet. 
You should clone H2Combustion from this repository:

    git clone https://github.com/THGLab/H2Combustion.git

I recommend using conda environment to install dependencies of this library. 
Please install (or load) conda and then proceed with the following commands:

    conda create --name torch-gpu python=3.7 
    conda activate torch-gpu
    conda install -c conda-forge numpy scipy pandas ase 
    conda install -c pytorch pytorch torchvision cudatoolkit=10.1
    
Now, you can change directory to your local H2Combustion repo and run modules in the `torch-gpu` environment.
The `script.voxelnet.py` uses implemented modules to run a voxel representaion based on the 
MR3Ddensenet model. Please modify parameters inside the script before run.

## Guidelines
- Please push your changes to a new branch and avoid merging with the master branch unless 
your work is reviewed by at least one other contributor.

- The documentation of the modules are available at most cases. Please look up local classes or functions 
and consult with the docstrings in the code.