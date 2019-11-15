#!/bin/sh

rm -rf memfree.txt

start=$(date +%s)

#bash run_single.sh 64 input_64.txt 63 63
bash run_single.sh 64 newInput.txt 63 63

end=$(date +%s)
time=$(( $end - $start ))
echo "runing time: $time s"

#mprof run -T 1 matrix.py & \
#bash mem-python.sh 1 60 >> memPython.txt
