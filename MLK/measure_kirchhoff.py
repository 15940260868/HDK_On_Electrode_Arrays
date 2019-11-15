#!/usr/bin/env python

from sys import stdout
import sys
import numpy as np 
import re
import pandas as pd
import os

r2 = np.array(
    [
        [1.0, 2.0], 
        [3.0, 4.0], 
    ]
)

r3 = np.array(
    [
        [1.0, 1.0, 1.0], 
        [1.0, 1.0, 1.0], 
        [1.0, 1.0, 1.0], 
    ]
)

# Convert a general equation into ij-valued:
def fill_index(i, j, k, m, n, eq):
    res = eq

    # figure out how to map k (m) into [1..n-1] for Ua's and Ub's
    if k != 0:
        k_map_val = k
        if k > j:
            k_map_val = k - 1
        res = res.replace('Ua_k', 'x['+str(k_map_val - 1 + n - 1)+']') # Assign x[n-1 .. 2n-3] to all Ua's 
    if m != 0:
        m_map_val = m
        if m > i:
            m_map_val = m - 1
        res = res.replace('Ub_m', 'x['+str(m_map_val - 1)+']') # Assign x[0 .. n-2] to all Ub's 

    #for U's and R's
    res = res.replace('_k', '['+str(k)+']')
    res = res.replace('_m', '['+str(m)+']')
    
    return res

# Generate the equations between the i-th row and the j-th column
# Each i-j combination yields 2N equations
# For this particular, index starts at 1 (for both i and j)
def eqs_gen_ij(i, j, n):

    res = []

    # Left (n-1) eqs:
    for k in range(1, n+1):
        if k != j: 
            eq = '(1 - Ua_k) / ' + str(r[i-1][k-1]) # hardcode the original voltage as 1       
            for m in range(1, n+1):
                if m != i:
                    eq += ' - (Ua_k - Ub_m) / ' + str(r[m-1][k-1])
                    eq = fill_index(i, j, k, m, n, eq)
            res += [eq]
    # Right (n-1) eqs:
    for m in range(1, n+1):
        if m != i: 
            eq = 'Ub_m / ' + str(r[m-1][j-1])        
            for k in range(1, n+1):
                if k != j:
                    eq += ' - (Ua_k - Ub_m) / ' + str(r[m-1][k-1])
                    eq = fill_index(i, j, k, m, n, eq)
            res += [eq]

    return res

# Generate the function and persist it to a Python file
def eqs_func(eqs, i, j):
    eqs_path = "./__eqs"
    if not os.path.exists(eqs_path):
        os.makedirs(eqs_path)
    f = open(eqs_path+"/eqs_func_"+str(i)+"_"+str(j)+".py", "w")
    f.write("def func(x):\n")
    f.write("\treturn [")
    for idx in range(len(eqs)):
        f.write("\n\t\t" + eqs[idx])
        if idx < len(eqs) - 1:
            f.write(",")
    f.write("\n\t]\n")
    f.close()

# Entry point
if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: " + sys.argv[0] + " <n> <file name> <i> <j>")
        exit(0)

    n = int(sys.argv[1])
    fname = sys.argv[2]
    i = int(sys.argv[3])
    if i < 0 or i >= n:
        print("i out of bound")
        exit(0)
    j = int(sys.argv[4])
    if i < 0 or i >= n:
        print("j out of bound")
        exit(0)

    df = pd.read_csv(fname, sep=" ", header=None) ## *********** changed ***********
    r = df.as_matrix()
    if n != len(r):
        stdout.write("Warning: rank n %d and input array length %d do not match.\n" % (n, len(r)))
        exit(0)

    eqs = eqs_gen_ij(i+1,j+1,n)
    eqs_func(eqs, i, j)

