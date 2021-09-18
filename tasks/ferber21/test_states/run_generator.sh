#!/bin/bash
RANDOM_SEED_LENGTH=$RANDOM$((RANDOM / 10))
RANDOM_SEED_WALK=$RANDOM$((RANDOM / 10))
SOURCE_PDDL=${1:-"$(dirname $0)/source.pddl"}

echo 'set $DEEPDOWN to the main directory of the code'
echo 'set $PDDL_GENERATORS to this directory'
exit 1
$DEEPDOWN/fast-downward.py $SOURCE_PDDL --translate-options --keep-unimportant-variables --search-options --search "generator(write_statistics=false,techniques=[iforward_none(50,distribution=uniform_int_dist(200,201,random_seed=$RANDOM_SEED_LENGTH),random_seed=$RANDOM_SEED_WALK)])" > /dev/null

for i in {0..49}; do
  $PDDL_GENERATORS/change_pddl_init_via_sas.py $SOURCE_PDDL "sas_plan${i}" > "p${i}.pddl"
  rm "sas_plan${i}"
done
rm output.sas

