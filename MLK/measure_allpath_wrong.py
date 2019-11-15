#!/usr/bin/env python

import sys
import numpy as np 

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

if __name__ == "__main__":
    if __debug__:
        sys.stdout.write("Input resistors is\n %s\n" % (r))
        i = 1
        j = 1
        sys.stdout.write("Input resistor r[%d][%d] is %s\n" % (i, j, r[i][j]))

#DFS of all paths
    """Depth-first-search for MEA

    Args:
        src (int): source X-axis, range(n)
        des (int): destination Y-axis, range(n)
        i (int): current X-axis, range(n)
        j (int): current Y-axis, range(n)
        n (int): the rank
        step (int): current progress toward n, range(n)
        z_local (int): the local aggregate z on the path
        z_total (int[]): the grant aggregate z, passed by reference

    Returns:
        none
    """
def step_to_dest (src, des, i, j, n, step, z_local, z_total):
    if step == n+1: # so we reach final X-axis
        if j == des: # we only count the measured Y-axis
            if __debug__:
                sys.stdout.write("i=%d, j=%d, step=%d, z_local=%s, z_total=%s\n" % (i, j, step, z_local, z_total))
            z_total[0] += 1 / z_local
    else: # or, let's move to the next X-axis
        for j_new in range(n):
            z_local_new = z_local
            if step == 1: # starting X-axis
                if __debug__:
                    sys.stdout.write("i=%d, j=%d, step=%d, z_local=%s, z_total=%s\n" % (i, j, step, z_local, z_total))
                z_local_new += r[i][j_new] # only downward
            elif j_new != j: # if we need to move to antoher Y-axis
                if __debug__:
                    sys.stdout.write("i=%d, j=%d, step=%d, z_local=%s, z_total=%s\n" % (i, j, step, z_local, z_total))
                z_local_new += r[i][j] + r[i][j_new] # need to upward and downward
            step_to_dest(src, des, (i+1)%n, j_new, n, step+1, z_local_new, z_total) # recursion

r = r2
z_tot = [.0]
n = len(r)
i = 0
j = n - 1
step_to_dest(i, j, i, 0, n, 1, .0, z_tot)
sys.stdout.write("1 / z[%d][%d] = %f\n" % (i, j, z_tot[0]))

