#!/bin/bash
RANDOM_SEED_LENGTH=$RANDOM$((RANDOM / 10))
RANDOM_SEED_WALK=$RANDOM$((RANDOM / 10))
SOURCE_PDDL=${1:-"$(dirname $0)/source.pddl"}
STEPS=${2:-500}
STEPSPLUSONE=`python3 -c "print(${STEPS} + 1)"`
rm -f output.sas

$DEEPDOWN/fast-downward.py --translate $SOURCE_PDDL --translate-options --keep-unimportant-variables >/dev/null
~/bin/h2pre < output.sas > /dev/null
$DEEPDOWN/fast-downward.py output.sas --search "generator(techniques=[gbackward_none(1,distribution=uniform_int_dist(${STEPS},${STEPSPLUSONE},random_seed=$RANDOM_SEED_LENGTH),random_seed=$RANDOM_SEED_WALK,deprioritize_undoing_steps=true,wrap_partial_assignment=false,is_valid_walk=true,bias=hff())],write_statistics=false,write_sas=true)" >/dev/null
$PDDL_GENERATORS/change_pddl_init_via_sas.py $SOURCE_PDDL sas_plan0
rm sas_plan0
rm output.sas

