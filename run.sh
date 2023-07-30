#!/bin/bash

make

g++ -std=c++11 -stdlib=libc++ generateTrainData.cpp -o genData.o

./genData.o > Data.txt

rm genData.o

./xor_nn

make clean