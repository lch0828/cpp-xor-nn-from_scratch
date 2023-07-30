#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


class Dataset
{
public:
	Dataset(const string filename);
	~Dataset();
	bool isEof(void);

	unsigned getNextInputs(vector<double> &input);
	unsigned gettarget(vector<double> &target);

private:
	ifstream _file;
};


#endif //DATASET_H