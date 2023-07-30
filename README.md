# C++ XOR Neural Network Implementation

This is a C++ implementaiton of a neural netowrk that learns the CPR task.

## Run

Just simply run the shell script and it will run the program!

```bash
$ ./run.sh
```
1. Generates the training data
2. Train the neural network

### Or

First compile the ```.cpp``` files.
```
make
```

Then compile the data generation file.
```
g++ -std=c++11 -stdlib=libc++ generateTrainData.cpp -o genData.o
```

Then generate the data/
```
./genData.o > Data.txt
```
Finally Train the neural network
```
./xor_nn
```
