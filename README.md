# Neural Fast Downward
Neural Fast Downward is intended to help with generating training data for
classical planning domains, as well as, using machine learning techniques with
Fast Downward (especially, Tensorflow and PyTorch). 

Neural Fast Downward is a fork from Fast Downward. For more information (full
list of contributors, history, etc.), see [here](https://github.com/PatrickFerber/NeuralFastDownward).


## Features
### Sampling
Neural Fast Downward implement a plugin type that takes a task as input and
creates new states from its state space. Currently the following two techniques
are implemented:
- Random walks with progression from the initial state
- Random walk with regression from the goal state

Furthermore, a sampling engine is implemented that takes the above mentioned
plugins evaluates the sampled states (produces for every state a vector of
strings) and stores them to disk (in the future hopefully to named pipes). How 
the states are evaluated depends on the concrete implementation of the 
sampling engine. Examples are:
- writing the new states as SAS tasks to disk
- solving the states using a given search algorithm and storing the states along
  the plan
- store an estimate for a state by evaluating its n-step successors with a 
  given heuristic function and back propagate the minimum estimate (similar to a
  Bellman update).
  
If you are only interested in the sampling code, just work with the branch 
`sampling`. The branch `main` contains the sampling feature, as well as, the
features below.
  
### Policies
Neural Fast Downward has some simple support for policies in classical planning.
It does **not** implement a good policy, but it provides a Policy class which can
be extended. Currently, two simple policies which internally rely on a given heuristic and
a simple search engine which follows the choices of a policy are implemented.

### Neural Networks
Neural Fast Downward supports Tensorflow and PyTorch models. It implements an 
abstract neural network base class and implements a network subclass for
Tensorflow and PyTorch. If you want to support another ML library take 
inspiration by those classes.

Furthermore, a network heuristic and network policy are provided. Both are simple
wrappers that take an abstract network and take the network outputs as heuristic
resp. policy values.

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
   0.8.0`](https://github.com/bazelbuild/bazel/releases/tag/0.8.0) appropriate
   to your platform and run the shell script with the `--user` flag. Put the
   created `bin` to your $PATH.

2. Download [`tensorflow 1.5.0`](https://github.com/tensorflow/tensorflow/releases/tag/v1.5.0) and
   extract it to a directory named `tensorflow`. Also, rename the inserted
   directory to `tensorflow`, so you'll have the directory structure
   `tensorflow/tensorflow`. Considering you're in the root directory, run the
   following commands:
    * `tensorflow`
    * `./configure`
    * `bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so`

3. Download the matching Protobuf and Eigen3 versions: 
    * `mkdir tensorflow/contrib/makefile/downloads/eigen`
    * Download [`eigen-3.3.4`](https://gitlab.com/libeigen/eigen/-/releases/3.3.4) and extract its contents to the directory created above.
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
    * `cd` to the `tensorflow` directory we first created (where we extracted the downloaded `tensorflow` folder).
    * `cd ..`
    * `mkdir protofuf`
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
    * `cp tensorflow/bazel-bin/tensorflow/*.so lib`
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
