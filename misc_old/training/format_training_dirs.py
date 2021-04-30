#!/usr/bin/env python
"""
Use this script to format the template XXXX-XX-XX-training-Template/... files which can be used to start a new training iteration and run search experiments.
"""
import format_training_sampling_tools as tools

from datetime import datetime
import math
import os
import subprocess
import sys


REPO_BASE = tools.get_repo_base(__file__)
DIR_DATA = os.path.join(REPO_BASE, "data", "FixedWorlds", "opt")
FILE_TEMPLATE_TRAINING = os.path.join(
    REPO_BASE, "XXXX-XX-XX-training-Template", "run_training.sh")
FILE_TEMPLATE_SEARCH = os.path.join(
    REPO_BASE, "XXXX-XX-XX-training-Template", "experiment_template.py")
assert os.path.isfile(FILE_TEMPLATE_TRAINING)
assert os.path.isfile(FILE_TEMPLATE_SEARCH)


NETWORK_ADAPTIVE = (
    "keras_adp_mlp(tparams=ktparams(epochs=1000,loss=mean_squared_error,"
    "batch=100,balance=$BALANCED,optimizer=adam,callbacks="
    "[keras_model_checkpoint(val_loss,./checkpoint.ckp),"
    "keras_progress_checking(val_loss,100,2,False,True),"
    "keras_early_stopping(val_loss,0.01,100),"
    "keras_restart(2,stop_successful=True),"
    "keras_stoptimer(max_time=86400,per_training=False,prevent_reinit=True,"
    "timeout_as_failure=True)]),"
    "hidden=$HIDDEN,"
    "residual_layers=$RESIDUAL_LAYERS,"
    "batch_normalization=$BATCH_NORMALIZATION,"
    "output_units=$OUTPUT_UNITS,"
    "pseudo_output_units=$PSEUDO_OUTPUT_UNITS,ordinal_classification=$ORDINAL,"
    "bin_size=$SIZE_BIN,activation=$ACTIVATION,dropout=${DROPOUT_FLOAT},"
    "l2=${LTWO},x_fields=[current_state,goals],y_fields=[hplan],"
    "learner_formats=[hdf5,protobuf],graphdef=graphdef.txt,count_samples=True)")

NETWORK_FIX = (
    "keras_mlp(tparams=ktparams(epochs=1000,loss=mean_squared_error,"
    "batch=100,balance=$BALANCED,optimizer=adam,callbacks="
    "[keras_model_checkpoint(val_loss,./checkpoint.ckp),"
    "keras_progress_checking(val_loss,100,2,False,True),"
    "keras_early_stopping(val_loss,0.01,100),"
    "keras_restart(2,stop_successful=True),"
    "keras_stoptimer(max_time=86400,per_training=False,prevent_reinit=True,"
    "timeout_as_failure=True)]),"
    "hidden_layer_size=$HIDDEN_LAYER_SIZE,"
    "batch_normalization=$BATCH_NORMALIZATION,"
    "output_units=$OUTPUT_UNITS,"
    "ordinal_classification=$ORDINAL,"
    "bin_size=$SIZE_BIN,activation=$ACTIVATION,dropout=${DROPOUT_FLOAT},"
    "l2=${LTWO},x_fields=[current_state,goals],y_fields=[hplan],"
    "learner_formats=[hdf5,protobuf],graphdef=graphdef.txt,count_samples=True)")

def get_date():
    return datetime.today().strftime('%Y-%m-%d')


NONEABLE_PARAMETER = {"DIR_SEARCH_SCRIPTS"}
BASH_EXTENDABLE_PARAMETERS = {"NETWORK"}
DEFAULT_PARAMETERS = {
    "WRITE_TRAINING_SCRIPT": True,
    "WRITE_SEARCH_SCRIPT": True,

    # Training script parameters
    "TYPE": "ocls",
    "SAMPLE_TYPE": "inter",
    "BALANCED": False,
    "HIDDEN": 3,
    "RESIDUAL_LAYERS": "[]",
    "HIDDEN_LAYER_SIZE": "[]",
    "BATCH_NORMALIZATION": 0,
    "SIZE_BINS": 1,
    "ACTIVATION": "sigmoid",
    "DROPOUT_RATE": 0,
    "L2_WEIGHT": 0.0,
    "PRUNING": "off",
    "PSEUDO_OUTPUT_UNITS": -1,
    "DIRECTORY": DIR_DATA,
    "DIRECTORY_FILTER": ".*(%s).*" % "|".join(tools.DIRS_AAAI20_MAIN),
    "SKIP_FLAGS":  "--skip-if-trained --skip-if-flag --skip-if-running",
    "TRAINING_SAMPLE_RESTRICTION": "",
    "VALIDATION_SAMPLE_RESTRICTION": "",
    "TEST_SAMPLE_RESTRICTION": "",
    "PREFIX_SAMPLE_COUNT": "Kall",
    "PREFIX_MODIFIER": "",
    "TEACHER": "sat",
    "NETWORK": NETWORK_ADAPTIVE,

    # Search script parameters
    "UNARY_THRESHOLD": [0.01],
    "DIR_SEARCH_SCRIPTS": os.path.join(
        REPO_BASE, "misc", "experiments"),
    "SEARCH_SUFFIX": "",
}


def _validate_parameter_type(parameter, default, value):
    assert isinstance(value, type(default)), \
            "Invalid parameter type for %s: %s instead of %s" % \
            (parameter, str(type(value)), str(type(default)))
    if type(value) == list:
        for v in value:
            _validate_parameter_type(parameter, default[0], v)


def validate_parameter_value(parameter, default, value):
    assert value is not None or parameter in NONEABLE_PARAMETER
    _validate_parameter_type(parameter, default, value)

    if parameter == "TYPE":
        assert value in ["reg", "cls", "ocls"]
    elif parameter == "SAMPLE_TYPE":
        assert value in ["inter", "init", "plan"]
    elif parameter == "BALANCED":
        assert value in [True, False]
    elif parameter in ["HIDDEN", "DROPOUT_RATE", "L2_WEIGHT"]:
        assert value >= 0
    elif parameter in ["UNARY_THRESHOLD"]:
        assert value > 0
    elif parameter == "SIZE_BINS":
        assert value >= 1
    elif parameter == "PSEUDO_OUTPUT_UNITS":
        assert value > 0 or value == -1
    elif parameter == "ACTIVATION":
        assert value in ["sigmoid", "tanh", "relu"]
    elif parameter == "PRUNING":
        assert value in ["off", "inter", "intra", "intra_inter"]
    elif parameter in ["DIRECTORY", "DIR_SEARCH_SCRIPTS"]:
        assert os.path.isdir(value)
    elif parameter == "TEACHER":
        assert value in ["opt", "sat", "gbfs_ff", "lama", "astar_ipdb",
                         "wastar2_ipdb", "wastar5_ipdb"]
    elif parameter == "BATCH_NORMALIZATION":
        assert 0 <= value <= 2
    elif parameter in (
            ["DIRECTORY_FILTER", "PREFIX_SAMPLE_COUNT", "PREFIX_MODIFIER",
             "SKIP_FLAGS", "WRITE_TRAINING_SCRIPT", "WRITE_SEARCH_SCRIPT",
             "SEARCH_SUFFIX", "NETWORK", "RESIDUAL_LAYERS",
             "HIDDEN_LAYER_SIZE"] +
            ["%s_SAMPLE_RESTRICTION" % k for k in
             ["TRAINING", "VALIDATION", "TEST"]]):
        pass
    else:
        assert False, "Unknown parameter: %s" % parameter


def __init_validate_default_parameter_values():
    for k, v in DEFAULT_PARAMETERS.items():
        validate_parameter_value(k, v, v)


__init_validate_default_parameter_values()


def load_template_training():
    with open(FILE_TEMPLATE_TRAINING, "r") as f:
        return f.read()


def load_template_search():
    with open(FILE_TEMPLATE_SEARCH, "r") as f:
        return f.read()


def convert_short_scientific(v, decimals=1):
    if v == 0:
        return v
    negative = "-" if v < 0 else ""
    v = abs(v)
    exp = math.floor(math.log10(v))
    base = v/(10.0**exp)
    if 0 <= exp < 2:
        return ('%%s%%.%if' % decimals) % (negative, base*10.0**exp)
    else:
        return ('%%s%%.%ife%%i' % decimals) % (negative, base, exp)


def string_char(k):
    return '"' if k in BASH_EXTENDABLE_PARAMETERS else "'"


def get_directory_name(parameters):
    dirname = ("{DATE}-training-{TYPE}{PSEUDO}{BINS}Ns{BALANCED}H{HIDDEN}"
               "{ACTIVATION}{SAMPLE_SELECT}Gen{TEACHER}Drp{DROPOUT_RATE}{L2}"
               "{PREFIX_SAMPLE_COUNT}Prune{PRUNING}{PREFIX_MODIFIER}").format(
        DATE=get_date(),
        TYPE=parameters["TYPE"].capitalize(),
        PSEUDO=("" if parameters["PSEUDO_OUTPUT_UNITS"] == -1
                else parameters["PSEUDO_OUTPUT_UNITS"]),
        BINS=("" if parameters["SIZE_BINS"] == 1
              else "Bins%i" % parameters["SIZE_BINS"]),
        BALANCED="Bal" if parameters["BALANCED"] else "Ubal",
        HIDDEN=parameters["HIDDEN"],
        ACTIVATION=parameters["ACTIVATION"].capitalize(),
        SAMPLE_SELECT=parameters["SAMPLE_TYPE"].capitalize(),
        DROPOUT_RATE=parameters["DROPOUT_RATE"],
        L2=("" if parameters["L2_WEIGHT"] == 0 else
            "LT%s" % convert_short_scientific(parameters["L2_WEIGHT"], 0)),
        PREFIX_SAMPLE_COUNT=parameters["PREFIX_SAMPLE_COUNT"].capitalize(),
        PRUNING=parameters["PRUNING"].capitalize(),
        PREFIX_MODIFIER=("" if parameters["PREFIX_MODIFIER"] == ""
                         else ("_" + parameters["PREFIX_MODIFIER"])),
        TEACHER=parameters["TEACHER"].capitalize(),
    )
    return os.path.join(os.getcwd(), dirname)


def get_network_prefix(file_training):
    try:
        output = subprocess.check_output([file_training, "--dry"])
    except Exception:
        pass
    assert type(output) == str, type(output)
    return [x.strip() for x in output.split("\n") if x.strip() != ""][-1]


def get_search_script_name(parameters, network_prefix):
    suffix = parameters["SEARCH_SUFFIX"]
    return "{DATE}-{NETWORK_PREFIX}{SUFFIX}.py".format(
        DATE=get_date(),
        NETWORK_PREFIX=network_prefix,
        SUFFIX=("_" if suffix != "" else "") + suffix
    )


def get_parameters(**kwargs):
    assert all(k in DEFAULT_PARAMETERS for k in kwargs.keys()), \
        "A parameter given is unknown: %s" ", ".join(kwargs.keys())
    new_params = {k: v for k, v in DEFAULT_PARAMETERS.items()}
    for k, v in kwargs.items():
        validate_parameter_value(k, DEFAULT_PARAMETERS[k], v)
        new_params[k] = v
    return new_params


def get_all_parameter_settings():
    """Create your settings here"""
    return [
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     ACTIVATION="relu"),
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     DROPOUT_RATE=20),
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     DROPOUT_RATE=40),
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     L2_WEIGHT=0.1),
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     L2_WEIGHT=1.),
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     L2_WEIGHT=10.),
        #
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     HIDDEN=0),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     HIDDEN=1),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     HIDDEN=5),
        #
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     BALANCED=True),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     PRUNING="intra_inter"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     BALANCED=True,
        #     PRUNING="intra_inter"),
        #
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     SAMPLE_TYPE="init"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     SAMPLE_TYPE="plan"),
        #
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     TYPE="cls"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     TYPE="reg"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_GRID),
        #     DIRECTORY_FILTER="",
        #     TYPE="ocls",
        #     PSEUDO_OUTPUT_UNITS=1),

        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     TEACHER="astar_ipdb"),
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     TEACHER="wastar2_ipdb"),
        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     TEACHER="wastar5_ipdb"),

        # get_parameters(
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         tools.DIRS_AAAI20_MAIN_EXTENDED),
        #     TEACHER="lama"),
        # get_parameters(
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_BLOCKSWORLD,
        #     TEACHER="astar_ipdb"),
        # get_parameters(
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_BLOCKSWORLD,
        #     TEACHER="wastar2_ipdb"),
        # get_parameters(
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_BLOCKSWORLD,
        #     TEACHER="wastar5_ipdb"),

        # # Storage Table 1
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     TYPE="cls"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     TYPE="reg"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     TYPE="ocls",
        #     PSEUDO_OUTPUT_UNITS=1),
        #
        # # Storage Table 2 Left Side
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     HIDDEN=0),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     HIDDEN=1),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     HIDDEN=5),
        #
        # # Storage Table 2 Right Side
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     ACTIVATION="relu"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     DROPOUT_RATE=20),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     DROPOUT_RATE=40),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     L2_WEIGHT=0.1),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     L2_WEIGHT=1.),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     L2_WEIGHT=10.),

        # # Storage/Grid Table 3 Left Side (w/o random-state[#plans])
        # get_parameters(
        #     DIRECTORY=DIR_DATA,
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         [tools.DIR_GRID, tools.DIR_STORAGE]),
        #     SAMPLE_TYPE="init",
        #     TRAINING_SAMPLE_RESTRICTION="{restriction_file:sample_count_12gb:"
        #                                 "$FOLD$}",
        #     PREFIX_MODIFIER="fixed_sample_count"
        # ),
        # get_parameters(
        #     DIRECTORY=DIR_DATA,
        #     DIRECTORY_FILTER=".*(%s).*" % "|".join(
        #         [tools.DIR_GRID, tools.DIR_STORAGE]),
        #     SAMPLE_TYPE="plan",
        #     TRAINING_SAMPLE_RESTRICTION="{restriction_file:sample_count_12gb:"
        #                                 "$FOLD$}",
        #     PREFIX_MODIFIER="fixed_sample_count"
        # ),

        # # Storage Table 3 Right Side
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     BALANCED=True),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     PRUNING="intra_inter"),
        # get_parameters(
        #     DIRECTORY=os.path.join(DIR_DATA, tools.DIR_STORAGE),
        #     DIRECTORY_FILTER=".*%s.*" % tools.DIR_STORAGE,
        #     BALANCED=True,
        #     PRUNING="intra_inter"),

        # TODO Grid/Storage Table 3 random-state[#plans]
        # Grid/Storage Table 5

    # ] + [
    #     get_parameters(
    #         DIRECTORY=DIR_DATA,
    #         DIRECTORY_FILTER=".*(%s).*" % "|".join(
    #             [tools.DIR_GRID, tools.DIR_STORAGE]),
    #         TRAINING_SAMPLE_RESTRICTION="{multiply:{restriction_file:sample_"
    #                                     "count_12gb:$FOLD$}:%.3f}" % m,
    #         PREFIX_SAMPLE_COUNT="P%s" % p
    #     )
    #     for (p, m) in [("75", 0.75), ("50", 0.5), ("25", 0.25), ("25-1", 0.025)]
    # ]

        get_parameters(
            DIRECTORY=DIR_DATA,
            DIRECTORY_FILTER=".*(%s).*" % "|".join(
                tools.DIRS_AAAI20_MAIN_EXTENDED2),
            NETWORK=NETWORK_ADAPTIVE,
            HIDDEN=1,
            RESIDUAL_LAYERS="[keras_residual_block(hidden_layer_count=2)]",
            BATCH_NORMALIZATION=1,
            PREFIX_MODIFIER="rb1",
        ),
        get_parameters(
            DIRECTORY=DIR_DATA,
            DIRECTORY_FILTER=".*(%s).*" % "|".join(
                tools.DIRS_AAAI20_MAIN_EXTENDED2),
            NETWORK=NETWORK_ADAPTIVE,
            HIDDEN=1,
            RESIDUAL_LAYERS="[keras_residual_block(hidden_layer_count=2),"
                            "keras_residual_block(hidden_layer_count=2)]",
            BATCH_NORMALIZATION=1,
            PREFIX_MODIFIER="rb2",
        ),
        get_parameters(
            DIRECTORY=DIR_DATA,
            DIRECTORY_FILTER=".*(%s).*" % "|".join(
                tools.DIRS_AAAI20_MAIN_EXTENDED2),
            NETWORK=NETWORK_FIX,
            HIDDEN=1,
            HIDDEN_LAYER_SIZE=(
                "[-1,keras_residual_block(hidden_layer_count=2,"
                "hidden_layer_size=-1)]"),
            BATCH_NORMALIZATION=1,
            PREFIX_MODIFIER="rb1_notadaptive",
        ),
        get_parameters(
            DIRECTORY=DIR_DATA,
            DIRECTORY_FILTER=".*(%s).*" % "|".join(
                tools.DIRS_AAAI20_MAIN_EXTENDED2),
            NETWORK=NETWORK_FIX,
            HIDDEN=1,
            HIDDEN_LAYER_SIZE=(
                "[-1,keras_residual_block(hidden_layer_count=2,"
                "hidden_layer_size=-1),keras_residual_block("
                "hidden_layer_count=2,hidden_layer_size=-1)]"),
            BATCH_NORMALIZATION=1,
            PREFIX_MODIFIER="rb2_notadaptive",
        ),
        get_parameters(
            DIRECTORY=DIR_DATA,
            DIRECTORY_FILTER=".*(%s).*" % "|".join(
                tools.DIRS_AAAI20_MAIN_EXTENDED2),
            NETWORK=NETWORK_FIX,
            HIDDEN=2,
            HIDDEN_LAYER_SIZE=(
                "[5000,1000,"
                "keras_residual_block(hidden_layer_count=2,"
                "hidden_layer_size=1000),"
                "keras_residual_block("
                "hidden_layer_count=2,hidden_layer_size=1000),"
                "keras_residual_block(hidden_layer_count=2,"
                "hidden_layer_size=1000),"
                "keras_residual_block(hidden_layer_count=2,"
                "hidden_layer_size=1000)]"),
            BATCH_NORMALIZATION=1,
            PREFIX_MODIFIER="nature_reference",
        ),
    ]


def run():
    all_parameter_settings = get_all_parameter_settings()
    assert len(all_parameter_settings) > 0
    template_training = load_template_training()
    template_search = load_template_search()

    for parameter_setting in all_parameter_settings:
        write_training = parameter_setting["WRITE_TRAINING_SCRIPT"]
        write_search = parameter_setting["WRITE_SEARCH_SCRIPT"]
        if not (write_training or write_search):
            continue

        # Create directories to store files
        dir_training_script = get_directory_name(parameter_setting)
        dir_search_script = parameter_setting["DIR_SEARCH_SCRIPTS"]
        if dir_search_script is None:
            dir_search_script = dir_training_script
        for d in set(([dir_training_script] if write_training else []) +
                     ([dir_search_script if write_search else []])):
            # assert not os.path.isdir(d), d
            if not os.path.exists(d):
                os.makedirs(d)

        file_training = os.path.join(dir_training_script, "run_training.sh")
        if write_training:
            script_training = tools.partial_format(
                template_training, **{"FORMAT_%s" % k: "%s%s%s" % (
                    string_char(k), v, string_char(k))
                                      for k, v in parameter_setting.items()})
            script_training = script_training.replace("/home/", "/infai/")
            with open(file_training, 'w') as f:
                f.write(script_training)
            os.chmod(file_training, 0o755)

        if write_search:
            assert os.path.isfile(file_training), file_training
            network_prefix = get_network_prefix(file_training)
            file_search = os.path.join(
                dir_search_script,
                get_search_script_name(parameter_setting, network_prefix)
            )
            script_search = tools.partial_format(
                template_search,
                FORMAT_UNARY_THRESHOLDS="[%s]" % ", ".join(
                    str(x) for x in parameter_setting["UNARY_THRESHOLD"]),
                FORMAT_PREFIXES="[\"%s\"]" % network_prefix,
                FORMAT_FILTER_DOMAINS="\"%s\"" %
                                      parameter_setting["DIRECTORY_FILTER"],
                FORMAT_SKIP_TASKS="[]")
            with open(file_search, 'w') as f:
                f.write(script_search)
            os.chmod(file_search, 0o755)


if __name__ == "__main__":
    run()
