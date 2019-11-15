#!/bin/bash

if [ $# -gt 3 ];then
    args=("$@")
    N=${args[0]}
    FNAME=${args[1]}
    I=${args[2]}
    J=${args[3]}
else
    echo -e "Please provide <N> and <file name> <i> <j>\n"
    exit 1
fi

python -O measure_kirchhoff.py $N $FNAME $I $J
python -O eqs_sol.py $N $FNAME $I $J
