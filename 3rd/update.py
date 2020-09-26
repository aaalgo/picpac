#!/usr/bin/env python3
import sys
import subprocess as sp

MODULES = [
         ('https://github.com/pybind/pybind11.git',
         ['include', 'LICENSE']),
         ('https://github.com/fmtlib/fmt.git',
         ['include', 'LICENSE.rst']),
         ('https://github.com/gabime/spdlog.git',
         ['include', 'LICENSE']),
         ('https://github.com/xtensor-stack/xtensor.git',
         ['include', 'LICENSE']),
         ('https://github.com/xtensor-stack/xtl.git',
         ['include', 'LICENSE']),
         ('https://github.com/xtensor-stack/xtensor-python.git',
         ['include', 'LICENSE']),
         ('https://github.com/xtensor-stack/xtensor-blas.git',
         ['include', 'LICENSE']),
         ('https://github.com/nlohmann/json',
         ['single_include', 'LICENSE.MIT']),
]

def call (cmd):
    sp.check_call(cmd, shell=True)

call('mkdir -p tmp')
for url, moves in MODULES:
    bname = url.rsplit('/', 1)[-1].split('.')[0]
    call(f'rm -rf {bname}; mkdir -p {bname}')
    call(f'if [ ! -d tmp/{bname} ]; then git clone --depth 1 {url} tmp/{bname}; fi')
    call(f'cd tmp/{bname}; git show | head -n 1 | cut -f 2 -d " " | cut -b 1-16 > ../../{bname}/commit')
    for move in moves:
        call(f'mv tmp/{bname}/{move} {bname}/')
call('rm -rf tmp')

