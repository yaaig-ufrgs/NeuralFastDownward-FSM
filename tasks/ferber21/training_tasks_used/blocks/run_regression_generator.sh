#!/bin/bash
RANDOM_SEED_LENGTH=$RANDOM$((RANDOM / 10))
RANDOM_SEED_WALK=$RANDOM$((RANDOM / 10))
SOURCE_PDDL="$(dirname $0)/source.pddl"
$DEEPDOWN/fast-downward.py $SOURCE_PDDL --search "generator(write_statistics=false,techniques=[gbackward_none(1,distribution=uniform_int_dist(10,51,random_seed=$RANDOM_SEED_LENGTH),random_seed=$RANDOM_SEED_WALK)])" >/dev/null
$PDDL_GENERATORS/change_pddl_init_via_sas.py $SOURCE_PDDL sas_plan0
rm sas_plan0
rm output.sas

