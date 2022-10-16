#!/usr/bin/env bash

fd_root=../..

for statespace_file in ${fd_root}/tasks/experiments/statespaces/*_hstar; do
    echo $statespace_file $(cat $statespace_file | wc -l)
done
