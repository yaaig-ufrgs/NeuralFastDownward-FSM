This directory contains all files to perform the same experiments which 
have been performed in the paper "Neural Network Heuristics for 
Classical Planning: A Study of Hyperparameter Space"

To test if the compilation was successful execute the the script 
examples/run_search.sh i (i can be between 1 and 200)


The following files and directories are given:
	- README.md: This file contains information on how to compile this
	Fast-Downward version (it is a bit more tricky, because you have to
	compile Tensorflow to a C++ library).
	- fast-downward.py: Starts up Fast-Downward a tool to solve planning
	problems
	- fast-sample.py: A script to sample for a specific problem data. 
	(see argument parser object for more information)
	- fast-training.py: Starts training a neural network with some 
	previously parsed data (see argument parser object for more information)
	- build.py: ./build.py [debug64dynamic|release64dynamic] to build
	Fast-Downward (required for sampling and searching not for training).
	Only dynamic builds are supported (because the tensorflow C++ library
	is dynamically linked)
	- python_requirements.txt: file in the pip  requirement list format
	which list all required python modules.
	
	- driver: contains scripts for running the Fast-Downward components
	
	- examples:
		- call_training.sh: example call to start a training run. requires
		previously sampled data.
		- run_example_search.sh: runs an example search on the depot_p05 tests
		tasks. Provide as argument the integer 1-200 (which defines which
		task to search on)
		- run_search.sh: script to run a search for a NN trained by the
		example described below.
	- misc:
		- experiments: The experiments were run using Lab by Jendrik 
		Seipp (https://lab.readthedocs.io/en/latest/). An example 
		experiment is given.
		- slurm: contains scripts to submit all steps (sampling, training,
		searching) to a slurm controller (the tool on our server 
		infrastructure to manage compute jobs)
		- training:
			- analyse_data_distribution.py: given some data files, the h
			value distribution in the data files is analysed
			- calculate_domain_properties.py: calculates some properties
			for a given directory with task files. The most relevant
			part of these properties is which atoms in a task set are 
			unchangeable in all tasks.
		- convert_sample_format.py: Converts the format in which the 
		samples are stored on disk into another format (might be useful
		to save disk space if a different format is more efficient for 
		the tasks at hand)
		- format_training_dirs.py: uses the templates 
		XXXX-XX-XX-training-Template to create a script for training a neural network and a script to use the trained neural network for search
		
	- src:
		- translate: source code files for the Fast-Downward translator
		which converts PDDL Tasks into SAS Tasks
		- search: source code for the Fast-Downward search component
		- training: source code for training NN
	- tasks: contains the PDDL files for the tasks evaluated. The top level
	contains a directory for each domain of which tasks were evaluated. Those
	directories contain directories for each task evaluated. In this directory
	is the original task file as source.pddl, 200 test tasks as p1.pddl 
	to p200.pddl, domain_properties.json, atoms.json, and run_generator.sh
	(which creates new tasks via random walk from the original task)
	
	- tools: a directory containing some helper methods.
		
	- XXXX-XX-XX-training-Template: directory containing templates for
	training and evaluating the trained networks. To fill the template 
	take a look at misc/training/format_training_dirs.py
	
	
To get into the code, the easiest way might be to start by:
1. Select a task to test and put the task and domain file in a directory.
   Rename the task file to source.pddl 
   (for example take a simple blocksworld task)
2. copy run_generator.sh into the directory and generate some test 
   instances either by yourself with run_generator.sh or by calling
   ./examples/generate_tasks.sh (attention, in either way, you should
   check that your test instances are not duplicates of each other,
   this can happen if your state space is very small)
3. calculate the domain properties, but calling 
   ./misc/training/calculate_domain_properties.py DIRECTORY
4. Start sampling data. Use some example sample call from ./fast-sample.py, 
   e.g. ./fast-sample.py PROBLEM_IN_DIRECTORY -f NonStatic_A_01 --generator AUTO 10 -o "gzip(file={FirstProblem:-5}.test.data.gz)" --ignore-generator-errors --state-select PLAN
5. Train a neural network on this data with ./fast-training.py, e.g. adapt
./fast-training.py "keras_adp_mlp(tparams=ktparams(epochs=10,loss=mean_squared_error,batch=100,balance=False,optimizer=adam,callbacks=[keras_model_checkpoint(val_loss,./checkpoint.ckp),keras_stoptimer(max_time=120,per_training=False,prevent_reinit=True,timeout_as_failure=True)]),hidden=3,output_units=-2, ordinal_classification=true,bin_size=1,dropout=0,x_fields=[current_state,goals],y_fields=[hplan],learner_formats=[hdf5,protobuf],graphdef=graphdef.txt,count_samples=True)" --prefix test_ -d examples/task/ --skip --skip-if-running --maximum-data-memory 0.1gb -dp --format NonStatic_A_01 --input "gzip(suffix=.test.data.gz)" -o -n model --fields goals hplan current_state --global-minimum-samples-per-set 10
6. Use the Neural Network in a search. Run it similarly like done in 
   run_example.sh or use lab like shown in misc/experiments. For this
   tutorial you might use:
   run_search.sh MODEL_PB DOMAIN TASK


