#!/bin/bash

# Create the facts file from a source pddl.
# Usage: ./gen_facts_file.sh source_pddl

SOURCE_PDDL=$1
FACTS_FILE="${1::-5}_facts.txt"
SAS_FILE="output.sas"
PLAN_FILE="samples"

../../fast-downward.py --sas-file $SAS_FILE --plan-file $PLAN_FILE --build release $SOURCE_PDDL --search "sampling_search_yaaig(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()), techniques=[gbackward_yaaig(searches=1, samples_per_search=1)])"
ATOMS=$(sed "2q;d" $PLAN_FILE)
ATOMS=${ATOMS#"#<State>="}
ATOMS=${ATOMS%";"}

echo $ATOMS > $FACTS_FILE
rm $SAS_FILE $PLAN_FILE
