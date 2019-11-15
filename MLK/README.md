# MLK
Machine Learning-based Kirchhoff (MLK) Analysis of Multidimensional Data

# Dependencies:
## Recommended: Anaconda
## Required package: mpi4py, numpy, scipy, pandas
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
sudo apt-get install python-mpi4py

# To quick start:
## chmod +x start.py
## ./start.py 64 input_64.txt 32

# To run a single pair:
## chmod +x run.sh
## ./run_single.sh

# To run all [n X n] serially:
## chmod +x run_serial.py
## ./run_serial.py

# To run parallel training:
## chmod +x run_parallel.py
## mpiexec -n 32 ./run_parallel.py 64 input_64.txt

# To ignore checking in the temporary equation files:
## add "__eqs/" to the .gitignore file
## add "__output/" to the .gitignore file
