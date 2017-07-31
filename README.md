# Pyculib\_sorting

Pyculib\_sorting provides simplified interfaces to CUDA sorting libraries.
At present it contains a wrapper around:

 * A radix sort implementation from [CUB](http://nvlabs.github.com/cub).
 * A segmented sort implementation from
   [ModernGPU](http://nvlabs.github.io/moderngpu)

Pyculib\_sorting is predominantly used by [Pyculib](https://github.com/numba/pyculib) to provide
sorting routines.


## Requirements
Pyculib\_sorting requires the following programs to build and test:
 * Python
 * NVIDIA's `nvcc` compiler

and the following Python packages
 * pytest
 * Numba


## Obtaining the source code
Pyculib\_sorting relies on git submodules to access the CUB and ModernGPU source code,
to obtain a code base suitable for building the libraries run:

```
#> git clone https://github.com/numba/pyculib_sorting.git

#> cd pyculib_sorting

#> git submodule update --init
```

the URL above may be adjusted to use `ssh` based
`git@github.com:numba/pyculib_sorting.git` as desired.


## Building the libraries

To build the libraries run:
```
#> python build_sorting_libs.py
```


## Testing

Testing uses pytest and is simply invoked with:
```
#> pytest
```


## Conda build

To create a conda package of Pyculib\_sorting, assuming conda-build is
installed, run:

```
#> conda build condarecipe
```

from the root directory of Pyculib\_sorting.

