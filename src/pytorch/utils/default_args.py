# Train
OUTPUT_LAYER = "regression"
NUM_FOLDS = 1
HIDDEN_LAYERS = 2
HIDDEN_UNITS = [250]
BATCH_SIZE = 64
ACTIVATION = "relu"
WEIGHT_DECAY = 0
DROPOUT_RATE = 0
SHUFFLE = True
SHUFFLE_SEED = -1
SAVE_BEST_EPOCH_MODEL = True
BIAS = True
TRAINING_SIZE = 0.8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 99999999
MAX_TRAINING_TIME = float("inf") # seconds
RANDOM_SEED = -1
LINEAR_OUTPUT = False
SCATTER_PLOT = False
SCATTER_PLOT_N_EPOCHS = -1
WEIGHTS_METHOD = "kaiming_uniform"
LOSS_FUNCTION = "mse"
COMPARED_HEURISTIC_CSV_DIR = ""
HSTAR_CSV_DIR = ""
BIAS_OUTPUT = True
NORMALIZE_OUTPUT = False
OUTPUT_FOLDER = "results"
MODEL = "resnet"
PATIENCE = 100
NUM_CORES = -1
SEED_INCREMENT_WHEN_BORN_DEAD = 100
CHECK_DEAD_ONCE = True
SAVE_HEURISTIC_PRED = False
FORCED_MAX_EPOCHS = 1000
RESTART_NO_CONV = True
DATALOADER_NUM_WORKERS = 0
SAMPLE_PERCENTAGE = 1.0
SWAP_SAMPLES_FROM = ""
UNIQUE_SAMPLES = False  # Considers both x and y
UNIQUE_STATES = False  # Only considers x
USE_GPU = False
POST_TRAIN_EVAL = False
ADDITIONAL_FOLDER_NAME = []

# Test
DOMAIN_PDDL = ""
SEARCH_ALGORITHM = "eager_greedy"
HEURISTIC = "nn"
UNARY_THRESHOLD = 0.01
MAX_SEARCH_TIME = float("inf")  # seconds
MAX_SEARCH_MEMORY = 4 * 1024  # MB
MAX_EXPANSIONS = float("inf")
TEST_MODEL = "all"
HEURISTIC_MULTIPLIER = 1
FACTS_FILE = ""
DEF_VALUES_FILE = ""
FORCED_MAX_SEARCH_TIME = 10 * 60  # seconds
AUTO_TASKS_N = 9999
AUTO_TASKS_FOLDER = "tasks"
AUTO_TASKS_SEED = 0
SAMPLES_FOLDER = None
SAVE_DOWNWARD_LOGS = False
SAVE_GIT_DIFF = False
UNIT_COST = False

# Evaluation
LOG_STATES = False
SAVE_PREDS = False
SAVE_PLOTS = True
FOLLOW_TRAIN = False
EVAL_SHUFFLE = False
EVAL_TRAINING_SIZE = 1.0
EVAL_SEED = -1
EVAL_SHUFFLE_SEED = -1

# Experiment
EXP_TYPE = "all" # ['single', 'all', 'combined']
EXP_CORES = 10
EXP_NET_SEED = "0..9"
EXP_SAMPLE_SEED = "0..9"
EXP_ONLY_TRAIN = False
EXP_ONLY_TEST = False
EXP_ONLY_EVAL = False
EXP_TRAINED_MODEL = ""
EXP_EVAL_SAMPLE = ""

# Sampling
SAMPLE_STATESPACE = ""
SAMPLE_TEST_TASKS_DIR = ""
SAMPLE_METHOD = "yaaig"
SAMPLE_TECHNIQUE = "rw" # ['rw', 'dfs', 'bfs', 'bfs_rw']
SAMPLE_STATE_REPRESENTATION = "complete" # ['complete', 'complete_nomutex', 'forward_statespace']
SAMPLE_MAX_SAMPLES = -1
SAMPLE_SEARCHES = -1
SAMPLE_PER_SEARCH = -1
SAMPLE_REGRESSION_DEPTH = "default" # [int, 'default', 'facts', 'facts_per_avg_effects']
SAMPLE_REGRESSION_DEPTH_MULTIPLIER = 1.0
SAMPLE_SEED = "0"
SAMPLE_MULT_SEEDS = "0"
SAMPLE_DIR = "samples"
SAMPLE_RANDOM_PERCENTAGE = 0.0 # [0.0 .. 1.0]
SAMPLE_RESTART_H_WHEN_GOAL_STATE = True
SAMPLE_STATE_FILTERING = "mutex" # ['none', 'mutex', 'statespace']
SAMPLE_BFS_PERCENTAGE = 0.1 # [0.0 .. 1.0]
SAMPLE_IMPROVEMENT = "none" # ['none', 'partial', 'complete', 'both']
SAMPLE_ALLOW_DUPLICATES = "all" # ['all', 'interrollout', 'none']
SAMPLE_SUI = 0 # ['0', '1']
SAMPLE_SUI_RULE = "supersets" # ['supersets', 'subsets', 'samesets']
SAMPLE_CORES = 1
SAMPLE_SEARCH_ALGORITHM = "greedy" # eager greedy
SAMPLE_SEARCH_HEURISTIC = "ff"
SAMPLE_K_DEPTH = 99999
SAMPLE_UNIT_COST = False
SAMPLE_MAX_TIME = -1
SAMPLE_MEM_LIMIT_MB = -1
SAMPLE_EVALUATOR = "blind()" # 'blind()' is equal to not apply. Use 'pdb(hstar_pattern([]))' for h*.
