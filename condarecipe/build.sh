#!/bin/bash

function build() {
  cd $RECIPE_DIR/..
  python build_sorting_libs.py
}

build
mkdir -p $PREFIX/lib

if [ `uname` == Linux ]
then
     EXT=so
fi

if [ `uname` == Darwin ]
then
    EXT=dylib
fi

cp $RECIPE_DIR/../lib/*.$EXT $PREFIX/lib
