import sys
from sys import stdout
from scipy.optimize import root
import numpy as np
import pandas as pd

# process arguments
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
n = int(sys.argv[1])
n_eqs = 2 * (n-1)

# dynamically load modules
module_name = "__eqs.eqs_func_%d_%d" % (i, j)
if __debug__:
    print(module_name)
exec("from %s import func" % module_name)

# solve the equations
guess = np.ones(n_eqs)
md = 'lm'
sol = root(func, guess, method=md)
if __debug__:
    print("len(sol.x) = " + str(len(sol.x)))
    print("sol.x = \n" + str(sol.x))

# calculate the overall resistence
volts = (sol.x)[0:n-1]
if __debug__:
    print("type(volts) = " + str(type(volts)))
volts = np.insert(volts, i, 1.0) # the original voltage is always 1.0 volt
if __debug__:
    print("volts = \n" + str(volts))
# df = pd.read_csv('input_64.txt', sep="\t", header=None)
df = pd.read_csv(fname, sep=" ", header=None)

r = (df.as_matrix())[:,j]
a_total = 0
for idx in range(n):
    a_total += volts[idx] / r[idx]
z_total = 1.0 / a_total
f = open("./__output/z_%d_%d" % (i, j), "w")
f.write("i=%d\tj=%d\t%f\n" % (i, j, z_total))
f.close()
