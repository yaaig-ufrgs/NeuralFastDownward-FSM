# Neural Fast Downward
Neural Fast Downward is intended to help with generating training data for
classical planning domains, as well as, using machine learning techniques with
Fast Downward (especially, Tensorflow and PyTorch). 

Neural Fast Downward is a fork from Fast Downward. For more information (full
list of contributors, history, etc.), see [here](https://github.com/PatrickFerber/NeuralFastDownward).

## Features
### Sampling
Neural Fast Downward generates data from a given task,
therefore, it uses **SamplingTechniques** which take a given task and modify
it and **SamplingEngines** which perform some action with the modified task.

**Current Sampling Techniques**:
- New initial state via random walks with progression from the original initial
  state
- New initial state via random walk with regression from the goal condition 

**Current Sampling Engines:**
- writing the new states as (partial) SAS tasks to disk
- use a given search algorithm to find plans for the new states and 
  store them
- estimate the heuristic value of a state by value update using the n-step 
  successors (like Bellman update).
  
If you are only interested in the sampling code, just work with the branch 
`sampling`. The branch `main` contains the sampling feature, as well as, the
features below.

**Example:**

Generate two state via regression from the goal with random walk lengths
 between 5 and 10. Use `A*(LMcut)` to find a solution and store all states
  along the plan, as well as the used operators. 

```./fast-downward.py --build BUILD ../benchmarks/gripper/prob01.pddl --search 
"sampling_search_simple(astar(lmcut(transform=sampling_transform()),transform=sampling_transform()), techniques=[gbackward_none(2, distribution=uniform_int_dist(5, 10))])"
```

*ATTENTION: By default, the components of Fast Downward (e.g. search engines 
and heuristics) use the original task. Thus, you have to provide them the
argument `transform=sampling_transform()`.*
  
[Click here for more information and examples](SAMPLING.md)
  
### Policies
Neural Fast Downward has some simple support for policies in classical planning.
It does **not** implement a good policy, but it provides a Policy class which can
be extended. Currently, two simple policies which internally rely on a given heuristic and
a simple search engine which follows the choices of a policy are implemented.

### Neural Networks
Neural Fast Downward supports Protobuf (Tensorflow 1.x) and PyTorch models. It 
implements an 
abstract neural network base class and implements subclass for
Tensorflow and PyTorch. Wrappers which use a NN to calculate a heuristic or
policy are implemented.

[Click here for more information and examples](NEURALNETWORKS.md)

Below are the build instructions for each ML framework, `PyTorch` and
`Tensorflow`. In case of error, open an issue.

0. Setup and activate a virtual environment with Python 3.6. Make sure you have `numpy`, `curl` and `cmake`.

#### PyTorch guide
1. Download [libtorch](https://pytorch.org/cppdocs/installing.html) and extract it to
any path `P`. 

2. Set an environment variable `PATH_TORCH` that points to `P`.

3. Uncomment the Torch Plugin in
`src/search/DownwardFiles.cmake`.

4. `./build.py`

5. `pip install torch`

6. If everything is working, you may be able to run the example on
   `examples/test_pytorch/`.

#### Tensorflow guide
The original authors don't guarantee that the Tensorflow code will work with
more recent versions of Tensorflow (2+). Therefore, we'll use Tensorflow 1.5.0
which was proved to work (however, it may possibly also work with 1.15.0).

1. Download the [`bazel
   0.8.0`](https://github.com/bazelbuild/bazel/releases/tag/0.8.0) installer appropriate
   to your platform and run the shell script with `--user`. Put the created `bin` to your `$PATH`.

2. Download [`tensorflow 1.5.0`](https://github.com/tensorflow/tensorflow/releases/tag/v1.5.0) and
   extract it to a newly created directory named `tensorflow`. Also, rename the extracted
   directory to `tensorflow`, so you'll have the directory structure
   `tensorflow/tensorflow`. Considering you're in the root directory
   `tensorflow`, run:
    * `cd tensorflow`
    * `./configure`
    * `bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so`

3. Download the matching Protobuf and Eigen3 versions: 
    * `mkdir tensorflow/contrib/makefile/downloads/eigen`
    * Download [`eigen-3.3.4`](https://gitlab.com/libeigen/eigen/-/releases/3.3.4) and extract its contents to the directory created above.
    * Comment some lines regarding Eigen (29: `EIGEN_URL` and 78: `download_and_extract`) in `./tensorflow/contrib/makefile/download_dependencies.sh`, as the script tries to download Eigen from a broken link.
    * Run `./tensorflow/contrib/makefile/download_dependencies.sh`. If it fails, try again -- the servers might be unstable.

4. Build Protobuf:
    * `cd tensorflow/contrib/makefile/downloads/protobuf/`
    * `mkdir /tmp/proto`
    * `./autogen.sh`
    * `./configure --prefix=/tmp/proto/`
    * `make`
    * `make install`

5. Build Eigen:
    * `cd ../eigen`
    * `mkdir /tmp/eigen`
    * `mkdir build_dir`
    * `cd build_dir`
    * `cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../`
    * `make install`

6. Create a library/include directory structure:

    At the end of this process, your directory structure will look like the
    following:
    ```
    P
    └───tensorflow/
    │   └───tensorflow/
    │   └───include/
    │   └───lib/
    │   
    └───protobuf/
    │   └───lib/
    │   └───bin/
    │   └───include/
    │   
    └───eigen/
    │   └───include/
    │   └───lib/
    ```
    * `cd P`
    * `mkdir protobuf`
    * `cd protobuf`
    * `mkdir include`
    * `mkdir lib`
    * `cd ..`
    * `mkdir eigen`
    * `cd eigen`
    * `mkdir include`
    * `mkdir lib`
    * `cd ..`
    * `cd tensorflow`
    * `mkdir lib`
    * `mkdir include`
    * `cp tensorflow/bazel-bin/tensorflow/*.so lib/`
    * `cp -r tensorflow/bazel-genfiles/* include/`
    * `cp -r tensorflow/third_party include/`
    * `cp -r tensorflow/tensorflow/contrib/makefile/downloads/nsync include/`
    * `cp -r -n tensorflow/tensorflow/core/* include/tensorflow/core/`
    * `cp -r /tmp/proto/* ../protobuf/`
    * `cp -r /tmp/eigen/include/eigen3/* ../eigen/include`
    * `cp -r /tmp/eigen/* ../eigen/`

7. Setup environment variables for each package to where you all the `include/` and `lib/` directories are stored:
    * `export PATH_TENSORFLOW=/absolute/path/to/tensorflow/`
    * `export PATH_PROTOBUF=/absolute/path/to/protobuf/`
    * `export PATH_EIGEN=/absolute/path/to/eigen/`

8. Build Neural Fast Downward :
    * After setting up Tensorflow, you have to uncomment the Tensorflow Plugin in `src/search/DownwardFiles.cmake`.
    * Finally, run `./build.py`.

9. To check if everything worked, run the example on
    `examples/test_tensorflow`.

* Trained models are available [here](https://zenodo.org/record/4000991).

[Click here for information on extending Neural Fast Downward](EXTENDING.md)

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
