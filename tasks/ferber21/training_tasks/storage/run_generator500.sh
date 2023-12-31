#!/bin/bash
RANDOM_SEED_LENGTH=$RANDOM$((RANDOM / 10))
RANDOM_SEED_WALK=$RANDOM$((RANDOM / 10))
SOURCE_PDDL=${1:-"$(dirname $0)/source.pddl"}

$DEEPDOWN/fast-downward.py $SOURCE_PDDL --translate-options --keep-unimportant-variables --search-options --search "generator(techniques=[iforward_none(1,distribution=uniform_int_dist(500,501,random_seed=$RANDOM_SEED_LENGTH),random_seed=$RANDOM_SEED_WALK)])" >/dev/null
$PDDL_GENERATORS/change_pddl_init_via_sas.py $SOURCE_PDDL sas_plan
rm sas_plan
rm output.sas

