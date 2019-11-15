#!/usr/bin/env python

import os, sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " <n>")
    exit(0)

n = int(sys.argv[1])
fname = 'output_%d.txt' % (n)

os.system('cat __output/* > %s' % fname)

f = open(fname, 'r')
lines = f.readlines()
res = np.zeros((n,n))
for ln in lines:
    vals = ln.split('\t')
    i = int(vals[0][2:])
    j = int(vals[1][2:])
    z = float(vals[2])
    res[i][j] = z
f.close()

np.savetxt(fname, res, fmt="%.6f", delimiter=",")
