#!/bin/bash

# usage: ./train-and-test.sh "train_args" "test_args"
#
# train_args = samples [-o {regression,prefix,one-hot}] [-f NUM_FOLDS] [-hl HIDDEN_LAYERS]
#              [-hu HIDDEN_UNITS [HIDDEN_UNITS ...]] [-b BATCH_SIZE] [-lr LEARNING_RATE]
#              [-e MAX_EPOCHS] [-t MAX_TRAINING_TIME] [-a {sigmoid,relu}] [-w WEIGHT_DECAY]
#              [-d DROPOUT_RATE] [-sh SHUFFLE] [-of OUTPUT_FOLDER] [-s SEED]
#
# test_args = train_folder domain_pddl problem_pddls [problem_pddls ...] [-a {astar,eager_greedy}]
#             [-heu {nn,add,blind,ff,goalcount,hmax,lmcut}] [-u UNARY_THRESHOLD] [-t MAX_SEARCH_TIME]
#             [-m MAX_SEARCH_MEMORY] [-e MAX_EXPANSIONS] [-pt {all,best}]

./train.py $1
./test.py $2
