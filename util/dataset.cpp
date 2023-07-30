#include"dataset.h"


Dataset::Dataset(const string filename)
{
	_file.open(filename.c_str());
}

Dataset::~Dataset() {}

bool Dataset::isEof(void)
{
    return _file.eof();
}

unsigned Dataset::getNextInputs(vector<double> &input)
{
    input.clear();

    string line;
    getline(_file, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double value;
        while (ss >> value) {
            input.push_back(value);
        }
    }

    return input.size();
}

unsigned Dataset::gettarget(vector<double> &target)
{
    target.clear();

    string line;
    getline(_file, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double value;
        while (ss >> value) {
            target.push_back(value);
        }
    }

    return target.size();
}