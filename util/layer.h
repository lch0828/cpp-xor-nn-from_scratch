#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "neuron.h"
#include <iostream>

using namespace std;

class Neuron;

class Layer;

class Layer
{
public:
	Layer(unsigned neuronNum);
	~Layer();

    void addNeuron(Neuron* neuron);

    void setPreviousLayer(Layer* layer);
    void setNextLayer(Layer* layer);

    Neuron* getNeuron(unsigned idx);
    unsigned getNeuronNum();

    void updateOutput();
    void calcGradient(const vector<double> &label, bool outputLayer);
    void updateWeights();

private:
    vector<Neuron*> _neurons;
    Layer* _previousLayer;
    Layer* _nextLayer;
    unsigned _neuronNum;

    static double lr;    // [0.0...1.0] learning rate
	static double alpha; // [0.0...n]   momentum
};


#endif // LAYER_H