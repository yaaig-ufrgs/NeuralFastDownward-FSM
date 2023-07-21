# NeuralFastDownward-FSM
- Code for the paper "Understanding Sample Generation Strategies for Learning Heuristic Functions in Classical Planning".

Neural Fast Downward is intended to help with generating training data for
classical planning domains, as well as, using machine learning techniques with
Fast Downward (especially, Tensorflow and PyTorch). 

NeuralFastDownward-FSM is a fork from [Ferber's Neural Fast Downward](https://github.com/PatrickFerber/NeuralFastDownward), which in turn derives from [Fast Downward](https://github.com/aibasel/downward).

Important: you can find our experiments from the paper in the `paper-experiments` directory.

## Fast Instructions

### Pre-run
1. Clone this repository.

2. Download and extract
   [`libtorch`](https://pytorch.org/cppdocs/installing.html) to a directory `p`.

3. `cd` to the directory where the root of the cloned repository is located, then:
    ```
    export Torch_DIR=p   OR   export PATH_TORCH=p
    pip install -r requirements.txt
    ./build.py release
    # And if interested in running FastDownward in debug mode:
    ./build.py debug
    ```
   3.1. If torch 1.9.0 is not found, install Python <= 3.9.10.

### Messing with the neural network code
See
[`src/pytorch/`](https://github.com/yaaig-ufrgs/NeuralFastDownward-FSM/tree/main/src/pytorch).

### Default arguments
See [`src/pytorch/utils/default_args.py`](https://github.com/yaaig-ufrgs/NeuralFastDownward-FSM/tree/main/src/pytorch/utils/default_args.py) and [`src/pytorch/utils/parse_args.py`](https://github.com/yaaig-ufrgs/NeuralFastDownward-FSM/tree/main/src/pytorch/utils/parse_args.py) for lists of default argument values when invoking programs.

### Generating samples

```
usage: fast-sample.py [-h] [-tst-dir TEST_TASKS_DIR] [-stp STATESPACE] [-tech {rw,dfs,bfs,bfs_rw}] [-search {greedy,astar}]
                      [-heur {ff,lmcut}] [-st {complete,complete_nomutex,forward_statespace}] [-max MAX_SAMPLES] [-scs SEARCHES]
                      [-sscs SAMPLES_PER_SEARCH] [-rd REGRESSION_DEPTH] [-rdm REGRESSION_DEPTH_MULTIPLIER] [-s SEED]
                      [-dups {all,interrollout,none}] [-ms MULT_SEED] [-c RANDOM_PERCENTAGE] [-rhg RESTART_H_WHEN_GOAL_STATE]
                      [-sf {none,mutex,statespace}] [-bfsp BFS_PERCENTAGE] [-o OUTPUT_DIR] [-sai {none,partial,complete,both}]
                      [-sui SUCCESSOR_IMPROVEMENT] [-suirule {supersets,subsets,samesets}] [-kd K_DEPTH] [-unit UNIT_COST]
                      [-cores CORES] [-t MAX_TIME] [-m MEM_LIMIT] [-eval EVALUATOR] [-dbg DEBUG]
                      instance {yaaig}
fast-sample.py: error: the following arguments are required: instance, method
```

The example below takes all the instances in the `blocks` directory and saves the
samples, facts and defaults files in the `samples` directory with an
appropriate filename. In the example below, we're generating 1000 samples. Of the final sample set, 500 are generated using BFS+RW and the remaining will be randomly generated. Duplicates are only allowed between rollout, states are completed with mutexes, all h-value improvements are used and the regression depth is limited by facts/avg(eff). 

```
./fast-sample.py tasks/experiments/blocks yaaig --technique bfs_rw --state-representation complete --max-samples 1000 --seed 0 --allow-dups interrollout --restart-h-when-goal-state yes --sample-improvement both --statespace tasks/experiments/statespaces/statespace_blocks_probBLOCKS-7-0_hstar --successor-improvement yes --regression-depth facts_per_avg_effects --state-filtering mutex --bfs-percentage 0.1 --random-percentage 0.5 --cores 1 --output-dir samples
```

### Training a neural network
Executing `./train.py -h` will show how to use it with all
the possible arguments. Almost everything is modifiable, and the default neural
network is a ResNet.

```
usage: train.py [-h] [-mdl {hnn,resnet}] [-sb SAVE_BEST_EPOCH_MODEL] [-diff SAVE_GIT_DIFF] [-pte POST_TRAIN_EVAL] [-pat PATIENCE]
                [-o {regression,prefix,one-hot}] [-lo LINEAR_OUTPUT] [-f NUM_FOLDS] [-hl HIDDEN_LAYERS]
                [-hu HIDDEN_UNITS [HIDDEN_UNITS ...]] [-b BATCH_SIZE] [-lr LEARNING_RATE] [-e MAX_EPOCHS] [-t MAX_TRAINING_TIME]
                [-a {sigmoid,relu,leakyrelu}] [-w WEIGHT_DECAY] [-d DROPOUT_RATE] [-shs SHUFFLE_SEED] [-sh SHUFFLE] [-gpu USE_GPU]
                [-bi BIAS] [-tsize TRAINING_SIZE] [-spt SAMPLE_PERCENTAGE] [-us UNIQUE_SAMPLES] [-ust UNIQUE_STATES]
                [-biout BIAS_OUTPUT] [-of OUTPUT_FOLDER] [-s SEED] [-sp SCATTER_PLOT] [-spn PLOT_N_EPOCHS]
                [-wm {default,sqrt_k,1,01,xavier_uniform,xavier_normal,kaiming_uniform,kaiming_normal,rai}] [-lf {mse,rmse}]
                [-no NORMALIZE_OUTPUT] [-rst RESTART_NO_CONV] [-cdead CHECK_DEAD_ONCE] [-sibd SEED_INCREMENT_WHEN_BORN_DEAD]
                [-trd NUM_CORES] [-dnw DATA_NUM_WORKERS] [-hpred SAVE_HEURISTIC_PRED]
                [-addfn [{patience,output-layer,num-folds,hidden-layers,hidden-units,batch-size,learning-rate,max-epochs,max-training-time,activation,weight-decay,dropout-rate,shuffle-seed,shuffle,use-gpu,bias,bias-output,normalize-output,restart-no-conv,sample-percentage,training-size} [{patience,output-layer,num-folds,hidden-layers,hidden-units,batch-size,learning-rate,max-epochs,max-training-time,activation,weight-decay,dropout-rate,shuffle-seed,shuffle,use-gpu,bias,bias-output,normalize-output,restart-no-conv,sample-percentage,training-size} ...]]]
                samples
```

The example below will train a neural network with a sampling file as input, utilizing seed 0 (for reproducibility), a max of 20 training epochs, ReLU activation, regression output, MSE loss function and Kaiming Uniform network initialization. The trained model will be saved in the `results` folder.

```
./train.py samples/yaaig_blocks_probBLOCKS-7-0_tech-bfsrw_sui_dups-ir_sai-both_repr-complete_bnd-factseff_maxs-1000_rs-500_ss0 -s 0 -e 20 -a relu -o regression -of results -lf mse -wm kaiming_uniform

```

### Evaluating instances
Executing `./test.py -h` will show how to use it with all
the possible arguments.

```
usage: test.py [-h] [-tfc TRAIN_FOLDER_COMPARE] [-diff SAVE_GIT_DIFF] [-d DOMAIN_PDDL] [-a {astar,eager_greedy}]
               [-heu {nn,add,blind,ff,goalcount,hmax,lmcut,hstar}] [-hm HEURISTIC_MULTIPLIER] [-u UNARY_THRESHOLD] [-t MAX_SEARCH_TIME]
               [-m MAX_SEARCH_MEMORY] [-e MAX_EXPANSIONS] [-pt {all,best,epochs}] [-sdir SAMPLES_DIR] [-ffile FACTS_FILE]
               [-dfile DEFAULTS_FILE] [-atn AUTO_TASKS_N] [-atf AUTO_TASKS_FOLDER] [-ats AUTO_TASKS_SEED] [-dlog DOWNWARD_LOGS]
               [-unit-cost UNIT_COST]
               train_folder [problem_pddls [problem_pddls ...]]
```

The example below takes a network folder (the trained model is located within
it) as the first argument and will automatically find 50 random (fixed seed as default) 
instances of the same domain to use for testing. `-t` is the time limit to solve the task, `-a` is the search algorithm used.

```
./test.py results/nfd_train.yaaig_blocks_probBLOCKS-7-0_tech-bfsrw_sui_dups-ir_sai-both_repr-complete_bnd-factseff_maxs-1000_rs-500_ss0.ns0 -t 360 -a eager_greedy
```

### Running full experiments
You can create multiple files like `exp_example.json` and call `./run.py exp_example.json`. Batch experiments will be performed according to the content in the JSON files. All the empty/unspecified settings will be run as
default, and missing sections will be ignored. 

You can find a multitude of examples in the `paper-experiments` directory.


## License

The following directory is not part of Fast Downward as covered by
this license:

- ./src/search/ext

For the rest, the following license applies:

```
Fast Downward is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

Fast Downward is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```
>>>>>>> origin/release
