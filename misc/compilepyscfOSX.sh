#!/bin/bash
cd #PATH/TO/PYSCF

export CMAKE_C_COMPILER=/usr/bin/cc
export CMAKE_CXX_COMPILER=/usr/bin/c++
export CC=/usr/bin/cc
export CXX=/usr/bin/c++
export CPP=/usr/bin/c++
export LD=/usr/bin/cc


rm -rf build
mkdir build
cd build
cmake ..
sed -i "" 's/-Wl,--no-as-needed//g' `grep -ril '\-Wl,--no-as-needed' .`
make
