#!/usr/bin/env python

from mpi4py import MPI
import sys, datetime, os

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

if len(sys.argv) < 3:
    print("Usage: " + sys.argv[0] + " <n> <file name>")
    exit(0)

n = int(sys.argv[1])
fname = sys.argv[2]

for i in range(n):
    for j in range(n):
        ind = i*n + j
        if rank == ind % size:
            if __debug__:
                os.system('echo rank=%d, i=%d, j=%d' % (rank, i, j))
            os.system('./run_single.sh %d %s %d %d' % (n, fname, i, j))
