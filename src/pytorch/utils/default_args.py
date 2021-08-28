# Train
DEFAULT_OUTPUT_LAYER = "regression"
DEFAULT_NUM_FOLDS = 1
DEFAULT_HIDDEN_LAYERS = 3
DEFAULT_HIDDEN_UNITS = []
DEFAULT_BATCH_SIZE = 100
DEFAULT_ACTIVATION = "sigmoid"
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_DROPOUT_RATE = 0
DEFAULT_SHUFFLE = False
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MAX_EPOCHS = 1000
DEFAULT_MAX_TRAINING_TIME = 86400 # seconds
DEFAULT_RANDOM_SEED = -1

# Test
DEFAULT_SEARCH_ALGORITHM = "astar"
DEFAULT_HEURISTIC = "nn"
DEFAULT_UNARY_THRESHOLD = 0.01
DEFAULT_MAX_SEARCH_TIME = 1800 # seconds
DEFAULT_MAX_SEARCH_MEMORY = 4000 # MB
DEFAULT_MAX_EXPANSIONS = float("inf")
DEFAULT_OUTPUT_FOLDER = "./results"
DEFAULT_TEST_MODEL = "all"
