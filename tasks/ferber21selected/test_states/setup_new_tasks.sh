#!/bin/sh

set -exu

BENCHMARKS="/home/ferber/repositories/benchmarks/"
DOMAIN=$(basename $(pwd))
DOMAIN_DIR="${BENCHMARKS}${DOMAIN}"
DOMAIN_FILE="${DOMAIN_DIR}/domain.pddl"
echo $DOMAIN_FILE

echo "Generate harder"
for task in `cat harder.txt`; do
  echo $task
  if [ ! -d "${task}" ]; then
    mkdir $task;
    cd $task;
    cp ${DOMAIN_FILE} "domain.pddl"
    ../../run_generator.sh "${DOMAIN_DIR}/${task}.pddl"
    cd ..
  else
    echo "Skip"
  fi
done
