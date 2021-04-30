#!/bin/sh
printf "\n\n\n%s\nRev: %s\n%s" "$(date)" "$(hg --debug id -i)" "$(cat "run_training.sh")" >> log.txt

## CHANGE ONLY THESE PARAMETERS
TYPE={FORMAT_TYPE}  # cls, reg, or ocls
SAMPLES={FORMAT_SAMPLE_TYPE}  # inter, plan, init
BALANCED={FORMAT_BALANCED}  #false or true
HIDDEN={FORMAT_HIDDEN}
HIDDEN_LAYER_SIZE={FORMAT_HIDDEN_LAYER_SIZE}  # use prefix modifier
SIZE_BIN={FORMAT_SIZE_BINS}  # >= 1
ACTIVATION={FORMAT_ACTIVATION}  # e.g. sigmoid, relu
DROPOUT={FORMAT_DROPOUT_RATE}  # use values from 0 to 100
RESIDUAL_LAYERS={FORMAT_RESIDUAL_LAYERS}  # E.G. "[]", use prefix modifier
BATCH_NORMALIZATION={FORMAT_BATCH_NORMALIZATION}  # 0
LTWO={FORMAT_L2_WEIGHT}  # enter l2 value as float
PRUNING={FORMAT_PRUNING}  #off, inter, intra, intra_inter
# this is used to tell keras_adp_mlp to build the architecture as
#if it would have X output units (the output layer contains neurons
#for all required units)
# -1 turns the feature off
PSEUDO_OUTPUT_UNITS={FORMAT_PSEUDO_OUTPUT_UNITS}  # -1 = OFF

DIRECTORY={FORMAT_DIRECTORY}  #"../../DeePDown/data/FixedWorlds/opt/"
# use "" for no filter, to filter for our default benchmarks use:
# ".*(blocksworld_fix_goals|blocksworld_ipc|depot_fix_goals|pipesworld-notankage_fix_goals|transport-opt14-strips|scanalyzer-08-strips|scanalyzer-opt11-strips).*"
DIRECTORY_FILTER={FORMAT_DIRECTORY_FILTER}
#DIRECTORY_FILTER=".*(grid_fix_goals|npuzzle_ipc|rovers|storage|visitall-opt14-strips).*"
TEACHER={FORMAT_TEACHER}  # teacher used to generate the data (e.g. opt/sat/lama/wastar2-ipdb)
SKIPS={FORMAT_SKIP_FLAGS}  #"--skip-if-trained --skip-if-flag --skip-if-running"

# Normally use PREFIX_MODIFIER if you add a sample restriction
TRAINING_SAMPLE_RESTRICTION={FORMAT_TRAINING_SAMPLE_RESTRICTION}  #""
VALIDATION_SAMPLE_RESTRICTION={FORMAT_VALIDATION_SAMPLE_RESTRICTION}  #""
TEST_SAMPLE_RESTRICTION={FORMAT_TEST_SAMPLE_RESTRICTION}  #""
PREFIX_SAMPLE_COUNT={FORMAT_PREFIX_SAMPLE_COUNT} # default 'Kall' if using all samples. KX for x samples, PX for X percent of total samples

PREFIX_MODIFIER={FORMAT_PREFIX_MODIFIER}  # e.g. use when adding training restriction


## EVERYTHING BELOW IS AUTOMATICALLY SET


if [ "$TYPE" = "cls" ]; then
	OUTPUT_UNITS=-2
	ORDINAL=false
elif [ "$TYPE" = "reg" ]; then
	OUTPUT_UNITS=-1
	ORDINAL=false
elif [ "$TYPE" = "ocls" ]; then
	OUTPUT_UNITS=-2
	ORDINAL=true
else
	echo "Invalid network type"  >&2
fi

if [ "$SAMPLES" = "inter" ]; then
	SAMPLE_ARGS="--sample-type plan --samples-per-problem 1"
elif [ "$SAMPLES" = "plan" ]; then
	SAMPLE_ARGS="--sample-type plan"
elif [ "$SAMPLES" = "init" ]; then
	SAMPLE_ARGS="--sample-type init"
else
	echo "Invalid sample type"  >&2
fi

if [ "$BALANCED" = "false" ] || [ "$BALANCED" = "False" ]; then
    PREFIX_BALANCED="ubal"
elif [ "$BALANCED" = "true" ] || [ "$BALANCED" = "True" ]; then
    PREFIX_BALANCED="bal"
else
	echo "Invalid balanced value"  >&2
fi

if [ ! "$DIRECTORY_FILTER" = "" ]; then
DIRECTORY_FILTER="-df $DIRECTORY_FILTER"
fi

if [ ! "$TRAINING_SAMPLE_RESTRICTION" = "" ]; then
TRAINING_SAMPLE_RESTRICTION="--samples-total-training '$TRAINING_SAMPLE_RESTRICTION'"
fi

if [ ! "$VALIDATION_SAMPLE_RESTRICTION" = "" ]; then
VALIDATION_SAMPLE_RESTRICTION="--samples-total-verifying '$VALIDATION_SAMPLE_RESTRICTION'"
fi

if [ ! "$TEST_SAMPLE_RESTRICTION" = "" ]; then
TEST_SAMPLE_RESTRICTION="--samples-total-testing '$TEST_SAMPLE_RESTRICTION'"
fi

if [ ! "$PSEUDO_OUTPUT_UNITS" = "-1" ]; then
    PREFIX_PSEUDO_OUTPUT_UNITS=$PSEUDO_OUTPUT_UNITS
else
    PREFIX_PSEUDO_OUTPUT_UNITS=""
fi

if [ ! "$PREFIX_MODIFIER" = "" ]; then
PREFIX_MODIFIER="${PREFIX_MODIFIER}_"
fi

if [ $SIZE_BIN -gt 1 ]; then
	PREFIX_SIZE_BIN="_bin$SIZE_BIN"
else
	PREFIX_SIZE_BIN=""
fi

if [ $BATCH_NORMALIZATION -eq 0 ]; then
  PREFIX_BATCH_NORMALIZATION=""
else
  PREFIX_BATCH_NORMALIZATION="_bn${BATCH_NORMALIZATION}"
fi

if [ "$(echo "$LTWO == 0" | bc -l)" -eq "1" ]; then
	PREFIX_LTWO=""
else
	SCIENTIFIC_LTWO=`python -c "import math
v=${LTWO}
decimals=0
if v == 0:
    print v
else:
    negative = '-' if v < 0 else ''
    v = abs(v)
    exp = math.floor(math.log10(v))
    base = v/(10.0**exp)
    if 0 <= exp < 2:
        print ('%%s%%.%if' % decimals) % (negative, base*10.0**exp)
    else:
        print ('%%s%%.%ife%%i' % decimals) % (negative, base, exp)"`

	PREFIX_LTWO="_LT${SCIENTIFIC_LTWO}"
fi

if [ "$PRUNING" = "off" ]; then
    PREFIX_PRUNING="Off"
elif [ "$PRUNING" = "inter" ]; then
    PREFIX_PRUNING="Inter"
elif [ "$PRUNING" = "intra" ]; then
    PREFIX_PRUNING="Intra"
elif [ "$PRUNING" = "intra_inter" ]; then
    PREFIX_PRUNING="IntraInter"
else
    echo "Invalid pruning type"
    exit 33
fi



DROPOUT_FLOAT=`python -c "print $DROPOUT / 100.0"`

NETWORK={FORMAT_NETWORK}  # NETWORK="keras_adp_mlp(tparams=ktparams(epochs=1000,loss=mean_squared_error,batch=100,balance=$BALANCED,optimizer=adam,callbacks=[keras_model_checkpoint(val_loss,./checkpoint.ckp),keras_progress_checking(val_loss,100,2,False,True),keras_early_stopping(val_loss,0.01,100),keras_restart(2,stop_successful=True),keras_stoptimer(max_time=86400,per_training=False,prevent_reinit=True,timeout_as_failure=True)]),hidden=$HIDDEN,output_units=$OUTPUT_UNITS,pseudo_output_units=$PSEUDO_OUTPUT_UNITS,ordinal_classification=$ORDINAL,bin_size=$SIZE_BIN,activation=$ACTIVATION,dropout=${DROPOUT_FLOAT},l2=${LTWO},x_fields=[current_state,goals],y_fields=[hplan],formats=[hdf5,protobuf],graphdef=graphdef.txt,count_samples=True)"

PREFIX="${TYPE}${PREFIX_PSEUDO_OUTPUT_UNITS}${PREFIX_SIZE_BIN}_ns_${PREFIX_BALANCED}_h${HIDDEN}_${ACTIVATION}${PREFIX_BATCH_NORMALIZATION}_${SAMPLES}_gen_${TEACHER}_drp${DROPOUT}${PREFIX_LTWO}_${PREFIX_SAMPLE_COUNT}_prune${PREFIX_PRUNING}_${PREFIX_MODIFIER}"

../fast-training.py $NETWORK --prefix $PREFIX -d $DIRECTORY $DIRECTORY_FILTER $TRAINING_SAMPLE_RESTRICTION $VALIDATION_SAMPLE_RESTRICTION $TEST_SAMPLE_RESTRICTION --samples-total-verifying 2000 --samples-total-testing 2000 -sdt --input "gzip(suffix=.generator.plan.${TEACHER}.data.gz)" --pruning $PRUNING --format NonStatic_A_01 -dp --slurm --cross-validation 10 -a "--export=ALL ../misc/slurm/slurm-training-15gb.sh" -o -n model --fields goals hplan current_state $SAMPLE_ARGS --plot-data-distribution --skip $SKIPS --maximum-data-memory 12GB --slurm-summarize $@

echo $PREFIX
