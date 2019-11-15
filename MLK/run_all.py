#!/usr/bin/env python

import sys
import pandas as pd
import subprocess
import time, datetime

if len(sys.argv) < 3:
    print("Usage: " + sys.argv[0] + " <n> <file name>")
    exit(0)

n = int(sys.argv[1])
fname = sys.argv[2]
df = pd.read_csv(fname, sep="\t", header=None)
r = df.as_matrix()
if n != len(r):
    stdout.write("Warning: rank n %d and input array length %d do not match.\n" % (n, len(r)))
    exit(0)

max_step = n

ts1 = time.time()

for i in range(max_step):
    for j in range(max_step):
        subprocess.call("./run.sh " + str(n) + " " + fname + " " + str(i) + " " + str(j), shell=True)

ts2 = time.time()

st = datetime.datetime.fromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S')
print("Start at: " + st)
st = datetime.datetime.fromtimestamp(ts2).strftime('%Y-%m-%d %H:%M:%S')
print("Finish at: " + st)
