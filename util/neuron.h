#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cmath>
#include "layer.h"

using namespace std;


class Neuron;

class Layer;

struct Edge
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned output_dim, unsigned idx);
	~Neuron();

	void setOutput(double val);
	double getOutput(void);

	void updateOutput(double sum);

	double getGradient(void);
	double getWeight(unsigned idx);
	double getDeltaWeight(unsigned idx);
	void updateWeight(unsigned idx, double deltaWeight);
	void setDeltaWeight(unsigned idx, double deltaWeight);

	void calcGradient(double inGradient);

private:
	static double activation(double x);
	static double activationDerivative(double x);

	static double randomWeight(void); // [0.0...1.0] randomWeight
	
	vector<Edge> _outputEdges;
	double _gradient;
	double _outputVal;
	unsigned _idx;

};

#endif // NEURON_H