# Running experiments

Unless stated otherwise, to perform each experiment, run the following in the root directory:

`./run.py tableX/*.json`

The generated samples and results will be saved in tableX/samples and tableX/results, respectively. 

Specifics are found within each tableX directory.

## Modifying JSONs

To easily change multiple JSONs at once in case you want to try different parameters, you can use the 
`modify_json.py` script. Example usage:

`./modify_json.py experiment exp-cores 5 tableX/*.json`

# Compiling results

To create a CSV with the data of an experiment, you can run:

`./make_csv.py tableX/results/*/nfd_train*/ > tableX/tableX.csv`

Afterwards, you can use your tool of choice to manipulate the data to manipulate the data.

* Table 3 (only for '-hmean' directories):
    - After generating the samples, run `./get_mean_h.py table3-algorithms-hmean/samples`

# TODO
- Table 8: Write script to evaluate the logic-based scripts to run the heuristics over the state spaces.
- Table 9: Using `run_heuristic_experiments.py`, write a specific hassle-free script to run the heuristics over the PDDLs.
