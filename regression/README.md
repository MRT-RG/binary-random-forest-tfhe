Regression using Binary Random Forest
================================================================================

This directory contains code for binary random forest classifier and
experiments on the MNIST dataset.

In this document, we assume that the users already finished the environment
construction. See [README.md](../README.md) for more details about the
environment construction.


Sample Usage
--------------------------------------------------------------------------------

You can train and test the binary random forest model on the Boston housing
dataset by the following command in this directory:

```console
make
```


Evaluation on the Boston Housing Dataset
--------------------------------------------------------------------------------

We have tried our binary random forest on the Boston housing dataset which is
a famouse dataset for a regression task. We've created the input binary vector
by quantizing the input real valued vector.

The following figure is a summary of our binary random forest regressor with
comparison to linear regressor. The linear regressor is one of commonly used
algorithm as a regressor on FHE scheme (in most cases CKKS scheme will be used).
The figure shows that our binary random forest achieves better performance
than linear regressor while the inference time is reasonable.

You can replicate our experiments by running `bash runall.bash`.

<p align="center">
  <img width="" src="../figures/regression_boston_housing.svg">
</p>


Implementation details


Our implementation is a kind of code generator, more precisely, our Python
script generate a CUDA source code which is compilable with `nvcc`. We can get
the final executable binary by compiling these CUDA code by `nvcc`. In this
section, we will explain about the procedure to generate a CUDA code and
compile it. The `Makefile` provides an example of the procedure for a fixed
hyperparameter.


Implementation details
--------------------------------------------------------------------------------
Our implementation is a kind of code generator, more precisely, our Python
script generates a CUDA source code that is compilable with `nvcc`. We can get
the final executable binary by compiling these CUDA codes by `nvcc`. In this
section, we will explain the procedure to generate a CUDA code and compile it.
The `Makefile` provides an example of the procedure for a fixed hyperparameter.

### (1) train.py
The following command train a random forest regressor model under the Boston
housing dataset on Python:

```console
python3 train.py
```

You can change the hyperparameters of the model like `max_dept`h and
`n_estimators` by command-line options. See `python3 train.py --help` for
more details. The above command will generate `model.pickle` (by default,
you can control the output file name by `--output` option) which contains
the number of input features and trained Scikit-learn's
`RandomForestRegressor` class instance.

**Note**: The number of input features is much larger than the number of
features in the original dataset (13 in the case of the Boston housing dataset)
because of the quantization. Our binary random forest can manage only binary
vector as an input, therefore, for manipulating the real-valued vector, we need
to quantize the real-valued vector. As a result of the quantization, the number
of input features increases than the original dataset. By default, the number of
features after quantization is 10 times larger than the number of features in
the original dataset, because the number of bins for the quantization is 10
(you can control the bins number by the `--bins` option).

This training code also dumps `testdata.txt` which contains quantized test data
as a text format. This file will be used when running the final executable
binary file.

**TODO**: The authors think the testdata.txt should contain the label information
in it, but not yet...

### (2) code_generator.py

The following command generates a CUDA code for the inference of our binary
random forest using the `model.pickle` which is created by the train.py (see the
previous section):

```console
python3 code_generator.py
```

Then the CUDA code will be generated and saved as `model.cu` by default.
The `model.cu` provides a CUDA function `prediction` and `get_result` that are
functions for running prediction and getting prediction results, respectively.
The main function (entry point of the CUDA code) is written in `main.cu`. You can
get the final executable binary by compiling `main.cu` and `model.cu` using `nvcc`
as described in the next subsection.

The generated code uses [cuFHE](https://github.com/vernamlab/cuFHE) library,
therefore, you need to build cuFHE before compiling the generated code.

### (3) main.cu & model.cu

The following command generates the final executable binary file which runs
inference of our binary random forest:

```console
nvcc -std=c++11 -O3 -IcuFHE/cufhe -LcuFHE/cufhe/bin -lcufhe_gpu -o runme main.cu model.cu
```

The above command will generate a binary file `runme`.

### (4) runme

Now you are ready to run the executable binary file! Please run the following
command:

```console
LD_LIBRARY_PATH=`pwd`/cuFHE/cufhe/bin ./runme `cat testdata.txt | head -n 1`
```

The binary file `runme` requires quantized input data as a 1st argument.
The quantized test data is dumped in `testdata.txt` by `train.py` and
the above command uses the 1st line of the file as an input.

By default, the executable binary will report the inference result and
average inference time.
