#!/usr/bin/env bash

abs_path=$(pwd)
fd_root=(${abs_path//\/NeuralFastDownward/ })
fd_root="${fd_root[0]}/NeuralFastDownward"

for statespace_file in ${fd_root}/tasks/experiments/statespaces/*_hstar; do
    echo $statespace_file $(cat $statespace_file | wc -l)
done
