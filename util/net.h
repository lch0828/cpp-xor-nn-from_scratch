#ifndef NET_H
#define NET_H

#include <vector>
#include <iostream>
#include "neuron.h"
#include "layer.h"

using namespace std;

class Net
{
public:
	Net(const vector<unsigned> &netShape);
	~Net();

	void feedForward(const vector<double> &inputVals);
	
	void calcError(const vector<double> &label);
	void calcGradients(const vector<double> &label);
	void updateWeights();
	void backProp(const vector<double> &label);

	void getResults(vector<double> &output);
	double getError() const;

private:
	vector<Layer*> _layers; //_layers[layerNum][neuronNum]
	double _error;
	double _recentAverageError;
	static double _recentAverageSmoothingFactor;
};


#endif // NET_H