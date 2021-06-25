from json import load

def load_training_state_value_tuples(json_file: str):
    """
    Load state-value pairs from a JSON file (Shen's format),
    returning a list of tuples in the format [([state], value)..].
    """
    with open(json_file,) as f:
        data = load(f)
        state_value_pairs = []

        for domain, training_pairs in data.items():
            for pair in training_pairs:
                state = [e[1:-1] for e in pair['state']]
                state_value = (state, pair['value'])
                state_value_pairs.append(state_value)

        return state_value_pairs

"""
Blocksworld state:

handempty
clear b1
on-table b3
on b1 b2
on b2 b3

Boolean format example for max blocks = 5

| handempty |   clear   |   holding   |   object  |  on-table |         on          |
-------------------------------------------------------------------------------------
| 1         | 1 0 0 0 0 |  0 0 0 0 0  | 0 0 0 0 0 | 0 0 1 0 0 | 1 0 0 1 0 0 0 0 0 0 |
"""
def states_to_boolean(data: [], domain: str):
    bool_states = []
    states = [t[0] for t in data]
    print(states)

    if domain == "blocksworld":
       bool_state = [0]*31 
    

# TODO - to avoid having to take the state-value pairs from Shen's data.
def generate_optimal_state_value_pairs(problem):
    pass

#data = load_training_state_value_tuples("domain_to_training_pairs_blocks.json")
#boolean_states = states_to_boolean(data, "blocksworld")
