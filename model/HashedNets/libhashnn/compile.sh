#! /bin/bash

buildDir="_build"

if [ ! -d "$buildDir" ]; then
  mkdir ${buildDir}
fi
cd ${buildDir}
cmake ..
make

cp libhashnn.so ../../




