#!/usr/bin/env python

import os, sys
import datetime
import time

def main(argv, output_name=None):
    # parse arguments  64 input_64.txt 32
    if len(argv) < 4:
        print("Usage: " + sys.argv[0] + " <n> <file name> <#threads>")
        exit(0)
    n = int(argv[1])
    fname = argv[2]
    k = int(argv[3])

    # prepare temp directories
    eqs_path = "__eqs"
    output_path = "__output"
    if not os.path.exists(eqs_path):
        os.system("cp __init__.py ./%s/" % eqs_path)
        os.makedirs(eqs_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # calculation
    os.system("mpiexec -n %d ./run_parallel.py %d %s" % (k, n, fname))

    # merge results
    os.system("./postprocess_output.py %d" % n)

    # rename output fname
    if output_name:
        fname = 'output_%d.txt' % (n)
        os.system("mv %s %s" % (fname, output_name))
    pass


if __name__ == '__main__':
    #if __name__ == '__main__':
    n = 64
    for i in range(83):
        fname = 'output_%d_%d.txt' % (n, i+317)
        main(['start.py', n, 'newInput_%d.txt' % (i+317), 32], fname)
    print("************* Finish*************")
    pass

'''
if __name__ == '__main__':
    coreNum = [64, 32, 16, 8, 4, 2]
    f = open('runtime.txt', 'a')
    for i in range(6):
        starttime = datetime.datetime.now()
        fname = 'output_%d_%d.txt' % (64, i)
        main(['start.py', 64, 'newInput.txt', coreNum[i]], fname)
        endtime = datetime.datetime.now()
        print("************* coreNum: {}, runTime: {}s *************".format(i, str((endtime - starttime).seconds)))
        f.writelines(str((endtime - starttime).seconds) + '\n')
    print("************* Finish*************")
    pass
'''
# if __name__ == '__main__':
#     main(['start.py', 64, 'input_64.txt', 32])
#     #main(sys.argv)
#     pass
