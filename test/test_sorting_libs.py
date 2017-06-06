from __future__ import print_function, absolute_import, division

"""
Uses radixsort implementation from CUB which has the following license:

Copyright (c) 2011, Duane Merrill.  All rights reserved.
Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
Uses segmented sort implementation from ModernGPU which has the following
license:

Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import ctypes
import os
import platform
import sys
import warnings
from contextlib import contextmanager

import pytest

import numpy as np
from numba import findlib
from numba.cuda.cudadrv.driver import device_pointer
from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.devicearray import auto_device, is_cuda_ndarray
from numba import cuda


def run_tests_on_hardware():
    def cuda_compatible():
        if sys.platform.startswith('darwin'):
            ver = platform.mac_ver()[0]
            # version string can contain two or three components
            major, minor = ver.split('.', 1)
            if '.' in minor:
                minor, micro = minor.split('.', 1)
            if (int(major), int(minor)) < (10, 9):
                return False

        is_64bits = sys.maxsize > 2**32
        if not is_64bits:
            return False

    return True

    if cuda_compatible():
        return cuda.is_available()
    else:
        return False


use_hardware = run_tests_on_hardware()


def library_extension():
    p = platform.system()
    if p == 'Linux':
        return 'so'
    if p == 'Windows':
        return 'dll'
    if p == 'Darwin':
        return 'dylib'


def load_lib(libname):
    fullname = 'pyculib_%s.%s' % (libname, library_extension())
    devpath = os.path.join(os.path.dirname(__file__), '..', 'lib')
    devlib = os.path.join(os.path.abspath(devpath), fullname)
    if os.path.exists(devlib):
        libpath = devlib
        warnings.warn('Using in-tree library %s' % libpath)
    else:
        libpath = os.path.join(findlib.get_lib_dir(), fullname)

    return ctypes.CDLL(libpath)


radixlib = load_lib('radixsort')
segsortlib = load_lib('segsort')


def _bind_radixsort_double():
    _argtypes = [
        ctypes.c_void_p,  # temp
        ctypes.c_uint,  # count
        ctypes.c_void_p,  # d_key
        ctypes.c_void_p,  # d_key_alt
        ctypes.c_void_p,  # d_vals
        ctypes.c_void_p,  # d_vals_alt
        cu_stream,
        ctypes.c_int,  # descending
        ctypes.c_uint,  # begin_bit
        ctypes.c_uint,  # end_bit
    ]
    dtype = np.float64
    fn = getattr(radixlib, "radixsort_double")
    fn.argtypes = _argtypes
    fn.restype = ctypes.c_void_p
    return fn


def test_radixsort_bind():
    # checks that the `radixsort_XYZ` symbols bind ok
    _known_types = ['float', 'double', 'int32', 'uint32', 'int64', 'uint64']
    for x in _known_types:
        getattr(radixlib, "radixsort_{}".format(x))


@pytest.mark.skipif(not use_hardware, reason='No suitable hardware found.')
def test_radixsort_operation():
    # a crude radixsort test
    dtype = np.float64
    maxcount = 1000

    keys = np.random.rand(maxcount)
    reference = np.copy(keys)

    # copy to device
    dptr, _ = auto_device(keys)

    def runsort(temp, keys, vals, begin_bit=0, end_bit=None):
        stream = 0
        begin_bit = 0
        dtty = np.dtype(dtype)
        end_bit = dtty.itemsize * 8
        descending = 0
        count = maxcount
        if keys:
            count = keys.size

        _arysize = int(maxcount * dtty.itemsize)
        _sort = _bind_radixsort_double()

        ctx = cuda.current_context()
        _temp_keys = ctx.memalloc(_arysize)

        return _sort(
            temp,
            ctypes.c_uint(count),
            device_pointer(keys),
            device_pointer(_temp_keys),
            None,
            None,
            stream,
            descending,
            begin_bit,
            end_bit
        )

    # tmp storage ref
    temp = runsort(None, None, None)

    # do the sort
    runsort(temp, dptr, None)

    # copy back
    dptr.copy_to_host(keys)

    # compare
    np.testing.assert_equal(np.sort(reference), keys)


def _bind_segsort_double():
    _argtypes = [
        ctypes.c_void_p,  # d_key
        ctypes.c_void_p,  # d_vals
        ctypes.c_uint,  # N
        ctypes.c_void_p,  # segments
        ctypes.c_uint,  # Nseg
        cu_stream,  # stream
    ]
    fn = getattr(segsortlib, 'segsortpairs_float64')
    fn.argtypes = _argtypes
    return fn


def test_segsort_bind():
    # checks that the `segsort_XYZ` symbols bind ok
    _known_types = ['float32', 'float64', 'int32', 'uint32', 'int64', 'uint64']
    for x in _known_types:
        getattr(segsortlib, "segsortpairs_{}".format(x))


@pytest.mark.skipif(not use_hardware, reason='No suitable hardware found.')
def test_segsort_operation():
    # a crude segsort test

    maxcount = 1000

    keys = np.random.rand(maxcount)
    reference = keys.copy()
    original = keys.copy()
    values = np.arange(keys.size, dtype=np.int32)
    segments = np.arange(64, maxcount, 64, dtype=np.int32)

    dptr_keys, _ = auto_device(keys)
    keys[:] = 0
    dptr_values, _ = auto_device(values)
    values[:] = 0
    dptr_segments, _ = auto_device(segments)

    def runsort(d_keys, d_vals, d_seg):
        _sort = _bind_segsort_double()
        _sort(device_pointer(d_keys),
              device_pointer(d_vals),
              d_keys.size,
              device_pointer(d_seg),
              d_seg.size,
              0)

    runsort(dptr_keys, dptr_values, dptr_segments)

    # copy back
    dptr_keys.copy_to_host(keys)
    dptr_values.copy_to_host(values)

    # compare
    r = [z for z in segments]
    low = [0] + r
    high = r + [maxcount]
    for x, y in zip(low, high):
        reference[x:y].sort()

    np.testing.assert_equal(keys, reference)
    np.testing.assert_equal(original[values], reference)
