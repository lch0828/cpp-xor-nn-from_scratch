#include "layer.h"

using namespace std;


double Layer::lr = 0.15; // 
double Layer::alpha = 0.5; //

Layer::Layer(unsigned neuronNum){ _neuronNum = neuronNum; }
Layer::~Layer(){}

void Layer::addNeuron(Neuron* neuron){ _neurons.push_back(neuron); }
void Layer::setPreviousLayer(Layer* layer){ _previousLayer = layer; }
void Layer::setNextLayer(Layer* layer){ _nextLayer = layer; }
Neuron* Layer::getNeuron(unsigned idx){ return _neurons[idx]; }
unsigned Layer::getNeuronNum(){ return _neuronNum; }

void Layer::updateOutput()
{
    for(unsigned n = 0; n < _neuronNum - 1; ++n)
    {
        double sum = 0.0;
        for(unsigned n_pre = 0; n_pre < _previousLayer->getNeuronNum(); ++n_pre)
	    {
		    sum += _previousLayer->getNeuron(n_pre)->getOutput() * _previousLayer->getNeuron(n_pre)->getWeight(n);
	    }
        _neurons[n]->updateOutput(sum);
    }
}

void Layer::calcGradient(const vector<double> &label, bool outputLayer)
{
    if(outputLayer)
    {
        for(unsigned n = 0; n < _neuronNum - 1; ++n)
        {
            double delta = label[n] - _neurons[n]->getOutput();
            _neurons[n]->calcGradient(delta);
        }

    }
    else 
    {
        for(unsigned n = 0; n < _neuronNum; ++n)
		{
            double sum = 0.0; // Sum our contributions of the errors at the nodes we feed
            for (unsigned n_next = 0; n_next < _nextLayer->getNeuronNum() - 1; ++n_next)
            {
                sum += _neurons[n]->getWeight(n_next) * _nextLayer->getNeuron(n_next)->getGradient();
            }
			_neurons[n]->calcGradient(sum);
		}
    }

}

void Layer::updateWeights()
{
    for(unsigned n = 0; n < _neuronNum - 1; ++n)
    {
        for(unsigned n_pre = 0; n_pre < _previousLayer->getNeuronNum(); ++n_pre)
	    {
           double oldDeltaWeight = _previousLayer->getNeuron(n_pre)->getDeltaWeight(n);
           double output = _previousLayer->getNeuron(n_pre)->getOutput();

           double newDeltaWeight = 
				lr * output * _neurons[n]->getGradient()
				+ alpha * oldDeltaWeight;

            _previousLayer->getNeuron(n_pre)->setDeltaWeight(n, newDeltaWeight);
		    _previousLayer->getNeuron(n_pre)->updateWeight(n, newDeltaWeight);
        }
    }
}



