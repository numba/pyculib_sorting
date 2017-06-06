# Pyculib\_sorting

Pyculib\_sorting provides simplified interfaces to CUDA sorting libraries.
At present it contains a wrapper around:

 * A radix sort implementation from [CUB](http://nvlabs.github.com/cub).
 * A segmented sort implementation from
   [ModernGPU](http://nvlabs.github.io/moderngpu)

Pyculib\_sorting is predominantly used by [Pyculib](INSERT_URL) to provide
sorting routines.


## Requirements
Pyculib\_sorting requires the following programs to build and test:
 * Python
 * Nvidia's `nvcc` compiler

and the following Python packages
 * pytest
 * numba


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

