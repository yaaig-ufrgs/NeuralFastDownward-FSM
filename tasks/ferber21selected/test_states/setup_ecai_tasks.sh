#!/bin/sh

set -exuo pipefail

if [[ $# -ne 2 ]]; then
  echo "USAGE: $0 DIR_WITH_TASK_AS_SUBDIRS PREFIX_FOR_TASK_IN_DIR"
  echo "Invalid arguments: Provide path to ecai domain repository (repo where for each task a directory is which contains the tasks"
  exit 2
fi

BENCHMARKS="/infai/ferber/repositories/benchmarks/"
DOMAIN=$(basename $(pwd))
DOMAIN_DIR="${BENCHMARKS}${DOMAIN}"
DOMAIN_FILE="${DOMAIN_DIR}/domain.pddl"
echo $DOMAIN_FILE

echo "Copy ECAI"
for task in `cat ecai.txt`; do
  echo $task
  mkdir $task;
  cp ${DOMAIN_FILE} "${task}/domain.pddl"
  for i in {1..50}; do
    ln -s "${1}/${2}${task}/p${i}.pddl" "${task}/p${i}.pddl"
  done
done
