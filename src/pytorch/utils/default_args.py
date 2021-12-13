# Train
DEFAULT_OUTPUT_LAYER = "regression"
DEFAULT_NUM_FOLDS = 1
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_HIDDEN_UNITS = [250]
DEFAULT_BATCH_SIZE = 64
DEFAULT_ACTIVATION = "relu"
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_DROPOUT_RATE = 0
DEFAULT_SHUFFLE = True
DEFAULT_SHUFFLE_SEED = -1
DEFAULT_BIAS = True
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_EPOCHS = 99999999
DEFAULT_MAX_TRAINING_TIME = float("inf") # seconds
DEFAULT_RANDOM_SEED = -1
DEFAULT_LINEAR_OUTPUT = False
DEFAULT_SCATTER_PLOT = True
DEFAULT_SCATTER_PLOT_N_EPOCHS = -1
DEFAULT_WEIGHTS_METHOD = "kaiming_uniform"
DEFAULT_WEIGHTS_SEED = -1
DEFAULT_COMPARED_HEURISTIC_CSV_DIR = ""
DEFAULT_HSTAR_CSV_DIR = ""
DEFAULT_BIAS_OUTPUT = True
DEFAULT_NORMALIZE_OUTPUT = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_MODEL = "resnet"
DEFAULT_PATIENCE = 100
DEFAULT_NUM_THREADS = -1
DEFAULT_SEED_INCREMENT_WHEN_BORN_DEAD = 100
DEFAULT_SAVE_HEURISTIC_PRED = False
DEFAULT_FORCED_MAX_EPOCHS = 1000
DEFAULT_RESTART_NO_CONV = True
DEFAULT_DATALOADER_NUM_WORKERS = 0
DEFAULT_CLAMPING = 0
DEFAULT_REMOVE_GOALS = False
DEFAULT_CONTRAST_FIRST = False
DEFAULT_STANDARD_FIRST = False
DEFAULT_INTERCALATE_SAMPLES = 0
DEFAULT_CUT_NON_INTERCALATED_SAMPLES = False
DEFAULT_SWAP_SAMPLES_FROM = ""

# Test
DEFAULT_DOMAIN_PDDL = ""
DEFAULT_SEARCH_ALGORITHM = "eager_greedy"
DEFAULT_HEURISTIC = "nn"
DEFAULT_UNARY_THRESHOLD = 0.01
DEFAULT_MAX_SEARCH_TIME = float("inf") # seconds
DEFAULT_MAX_SEARCH_MEMORY = 4*1024 # MB
DEFAULT_MAX_EXPANSIONS = float("inf")
DEFAULT_TEST_MODEL = "all"
DEFAULT_HEURISTIC_MULTIPLIER = 1
DEFAULT_FACTS_FILE = ""
DEFAULT_DEF_VALUES_FILE = ""
DEFAULT_FORCED_MAX_SEARCH_TIME = 10*60 # seconds
DEFAULT_AUTO_TASKS_N = 9999
DEFAULT_AUTO_TASKS_FOLDER = "tasks/ferber21/test_states"
DEFAULT_AUTO_TASKS_SEED = 0
DEFAULT_SAMPLES_FOLDER = None
DEFAULT_SAVE_DOWNWARD_LOGS = False

# Experiment
DEFAULT_EXP_TYPE = "all"
DEFAULT_EXP_THREADS = 10
DEFAULT_EXP_NET_SEED = 1
DEFAULT_EXP_SAMPLE_SEED = 1
DEFAULT_EXP_FIXED_SEED = 1
DEFAULT_EXP_ONLY_TRAIN = False
DEFAULT_EXP_ONLY_TEST = False

# Sampling
DEFAULT_SAMPLE_METHOD = "fukunaga"
DEFAULT_SAMPLE_TECHNIQUE = "rw"
DEFAULT_SAMPLE_STATE_REPRESENTATION = "fs"
DEFAULT_SAMPLE_SEARCHES = 500
DEFAULT_SAMPLES_PER_SEARCH = 200
DEFAULT_SAMPLE_SEED = 1
DEFAULT_SAMPLE_MULT_SEEDS = 1
DEFAULT_SAMPLE_DIR = "samples"
DEFAULT_SAMPLE_CONTRASTING = 50 # %
DEFAULT_SAMPLE_RESTART_H_WHEN_GOAL_STATE = True
DEFAULT_SAMPLE_MATCH_HEURISTICS = True
DEFAULT_SAMPLE_ASSIGNMENTS_US = 10
DEFAULT_SAMPLE_RSL_NUM_TRAIN_STATES = 10000
DEFAULT_SAMPLE_RSL_NUM_DEMOS = 5
DEFAULT_SAMPLE_RSL_MAX_LEN_DEMO = 500
DEFAULT_SAMPLE_RSL_STATE_INVARS = True
DEFAULT_SAMPLE_THREADS = 1
DEFAULT_SAMPLE_SEARCH_ALGORITHM = "greedy" # eager greedy
DEFAULT_SAMPLE_SEARCH_HEURISTIC = "ff"
DEFAULT_SAMPLE_FERBER_TECHNIQUE = "forward"
DEFAULT_SAMPLE_FERBER_MIN_WALK_LENGTH = "10"
DEFAULT_SAMPLE_FERBER_MAX_WALK_LENGTH = "20"
DEFAULT_SAMPLE_FERBER_SELECT_STATE = "random_state"
DEFAULT_SAMPLE_FERBER_NUM_TASKS = 50
