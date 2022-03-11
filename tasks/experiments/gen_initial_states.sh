#!/bin/bash

# Generates N initial states from a source pddl.
# Usage: ./gen_initial_stataes.sh source_pddl

N=50 # number of new initial states

SOURCE_PDDL=$1
DEST_FOLDER=${1::-5}

RANDOM_SEED_LENGTH=$RANDOM$((RANDOM / 10))
RANDOM_SEED_WALK=$RANDOM$((RANDOM / 10))

mkdir $DEST_FOLDER
cp $(dirname $1)/domain.pddl $DEST_FOLDER/domain.pddl

"../../fast-downward.py" $SOURCE_PDDL --translate-options --keep-unimportant-variables --search-options --search "generator(techniques=[iforward_none($N,distribution=uniform_int_dist(200,201,random_seed=$RANDOM_SEED_LENGTH),random_seed=$RANDOM_SEED_WALK)])" > /dev/null

for i in $(eval echo {0..$(($N-1))}); do
    ./change_pddl_init_via_sas.py $SOURCE_PDDL "sas_plan${i}" > "$DEST_FOLDER/p${i}.pddl"
    rm "sas_plan${i}"
done
