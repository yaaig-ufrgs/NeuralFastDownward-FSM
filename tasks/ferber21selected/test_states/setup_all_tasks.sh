#!/bin/sh

set -exuo pipefail

get_nb_items() { echo $#; }

BENCHMARKS="/infai/ferber/repositories/benchmarks/"
DOMAIN=$(basename $(pwd))
DOMAIN_DIR="${BENCHMARKS}${DOMAIN}"
DOMAIN_FILE="${DOMAIN_DIR}/domain.pddl"
echo $DOMAIN_FILE

echo "Generate all tasks"
for task in `ls "${DOMAIN_DIR}"| grep -v "domain"| grep ".pddl"`; do
  echo $task
  task_dir=${task::-5}
  mkdir $task_dir;
  cd $task_dir;
  if [[ -e "${DOMAIN_FILE}" ]]; then
    cp ${DOMAIN_FILE} "domain.pddl"
  else
    ALT_DOMAIN_FILES=`ls "${DOMAIN_DIR}" | grep "domain" | grep "${task_dir}"`
    echo "${ALT_DOMAIN_FILES}"
    if [[ ! "$(get_nb_items "${ALT_DOMAIN_FILES}")" -eq "1" ]]; then
      echo "err multiple possible domain files";
      exit 33;
    fi
    cp "${DOMAIN_DIR}/${ALT_DOMAIN_FILES}" "domain.pddl"
  fi

  ../../run_generator.sh "${DOMAIN_DIR}/${task}"
  cd ..
done
