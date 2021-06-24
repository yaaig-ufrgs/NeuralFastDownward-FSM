from json import load

JSON_FILE = "domain_to_training_pairs_blocks.json"

def load_training_data(json_file: str):
    """
    Load state-value pairs from a JSON file (Shen's format),
    returning a list of tuples in the format [([state], value)..].
    """
    with open(json_file,) as f:
        data = load(f)
        state_value_pairs = []

        for domain, training_pairs in data.items():
            for pair in training_pairs:
                state_value = (pair['state'], pair['value'])
                state_value_pairs.append(state_value)

        return state_value_pairs

# This will take time to let's focus on this later.
def generate_optimal_state_value_pairs(problem):
    pass
