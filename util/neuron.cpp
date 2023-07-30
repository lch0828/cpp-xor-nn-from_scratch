#include"neuron.h"
#include<iostream>


Neuron::Neuron(unsigned output_dim, unsigned idx)
{
	for(unsigned c = 0; c < output_dim; ++c)
	{
		_outputEdges.push_back(Edge());
		_outputEdges.back().weight = randomWeight();
	}

	_idx = idx;
}

Neuron::~Neuron(){}	
void Neuron::setOutput(double val) { _outputVal = val; }
double Neuron::getOutput(void) { return _outputVal; }
double Neuron::randomWeight(void) { return rand() / double(RAND_MAX); }
double Neuron::getGradient(void) { return _gradient; }
double Neuron::getWeight(unsigned idx) { return _outputEdges[idx].weight; }
double Neuron::getDeltaWeight(unsigned idx) { return _outputEdges[idx].deltaWeight; }
void Neuron::updateWeight(unsigned idx, double deltaWeight) { _outputEdges[idx].weight += deltaWeight; }
void Neuron::setDeltaWeight(unsigned idx, double deltaWeight) { _outputEdges[idx].deltaWeight = deltaWeight; }

void Neuron::updateOutput(double sum)
{
	_outputVal = Neuron::activation(sum);
}

void Neuron::calcGradient(double inGradient)
{
	_gradient = inGradient* Neuron::activationDerivative(_outputVal);
}

double Neuron::activation(double x)
{
	// tanh - output range [-1.0..1.0]
	return tanh(x);
}

double Neuron::activationDerivative(double x)
{
	// tanh derivative
	return 1.0 - x * x;
}
