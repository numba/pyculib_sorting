# A script to build external dependencies

import os
import subprocess
import platform


def basedir():
    return os.path.abspath(os.path.dirname(__file__))


def cub_include():
    return '-I%s/thirdparty/cub' % basedir()


def mgpu_include():
    return '-I%s/thirdparty/moderngpu/include' % basedir()


def lib_dir():
    return '%s/lib' % basedir()


def run_shell(cmd):
    print(cmd)
    subprocess.check_call(cmd, shell=True)


def library_extension():
    p = platform.system()
    if p == 'Linux':
        return 'so'
    if p == 'Windows':
        return 'dll'
    if p == 'Darwin':
        return 'dylib'


def gencode_flags():
    # Generate code for all known architectures
    GENCODE_SMXX = "-gencode arch=compute_{CC},code=sm_{CC}"
    GENCODE_SM20 = GENCODE_SMXX.format(CC=20)
    GENCODE_SM30 = GENCODE_SMXX.format(CC=30)
    GENCODE_SM35 = GENCODE_SMXX.format(CC=35)
    GENCODE_SM37 = GENCODE_SMXX.format(CC=37)
    GENCODE_SM50 = GENCODE_SMXX.format(CC=50)
    GENCODE_SM52 = GENCODE_SMXX.format(CC=52)
    GENCODE_SM53 = GENCODE_SMXX.format(CC=53)

    # Provide forward-compatibility to architectures beyond CC 5.3
    GENCODE_COMPUTEXX = "-gencode arch=compute_{CC},code=compute_{CC}"
    GENCODE_COMPUTE53 = GENCODE_COMPUTEXX.format(CC=53)

    # Concatenate flags
    SM = []
    SM.append(GENCODE_SM20)
    SM.append(GENCODE_SM30)
    SM.append(GENCODE_SM35)
    SM.append(GENCODE_SM37)
    SM.append(GENCODE_SM50)
    SM.append(GENCODE_SM52)
    SM.append(GENCODE_SM53)
    SM.append(GENCODE_COMPUTE53)
    return ' '.join(SM)


def build_cuda(srcdir, out, ins, includes):
    nvcc = locate_nvcc()

    # Build for 32- or 64-bit
    optflags = '-m%s --compiler-options "-fPIC"'
    if tuple.__itemsize__ == 4:
        opt = optflags % 32
    elif tuple.__itemsize__ == 8:
        opt = optflags % 64

    ext = library_extension()
    output = os.path.join(lib_dir(), '%s.%s' % (out, ext))
    inputs = ' '.join([os.path.join(srcdir, p)
                       for p in ins])
    argtemp = '{opt} {inc} -O3 {gen} --shared -o {out} {inp}'
    args = argtemp.format(inc=includes, gen=gencode_flags(), out=output,
                          inp=inputs, opt=opt)
    cmd = ' '.join([nvcc, args])
    run_shell(cmd)


def locate_nvcc():
    """
    Locate nvcc command on the platform, allowing specification
    of nvcc location in NVCC env var.

    """
    return os.environ.get('NVCC', 'nvcc')


def build_radixsort():
    build_cuda(srcdir=lib_dir(),
               out='pyculib_radixsort',
               ins=['cubradixsort.cu'],
               includes=cub_include(), )


def build_mgpusort():
    build_cuda(srcdir=lib_dir(),
               out='pyculib_segsort',
               ins=['mgpusort.cu'],
               includes=mgpu_include(), )


if __name__ == '__main__':
    build_radixsort()
    build_mgpusort()
