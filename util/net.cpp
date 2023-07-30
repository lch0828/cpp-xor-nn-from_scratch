#include"net.h"


double Net::_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

Net::Net(const vector<unsigned> &netShape)
{
	unsigned numLayers = netShape.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		Layer* layer = new Layer(netShape[layerNum] + 1);
		_layers.push_back(layer);
		Layer* previousLayer;
		
		// output_dim of last idx is 0
		unsigned output_dim = (layerNum == netShape.size() - 1 ? 0 : netShape[layerNum + 1]);

		// Fill layer with neurons + bias
		for(unsigned neuronNum = 0; neuronNum <= netShape[layerNum]; ++neuronNum)
		{
			Neuron *neuron = new Neuron(output_dim, neuronNum);
			_layers.back()->addNeuron(neuron);
		}

		// Bias output is 1.0
		_layers.back()->getNeuron(netShape[layerNum])->setOutput(1.0);

		// Set previous layer
		if(layerNum != 0)
		{
			_layers.back()->setPreviousLayer(previousLayer);

			// Set next layer
			previousLayer->setNextLayer(_layers.back());
		}
		previousLayer = _layers.back();
		cout<<"Layer["<<layerNum<<"]"<<" initialzied"<<endl;
	}
}

Net::~Net(){}

void Net::feedForward(const vector<double> &inputVals)
{
	// Check the num of inputVals euqal to neuronnum expect bias
	assert(inputVals.size() == _layers[0]->getNeuronNum() - 1);

	// Assign {latch} the input values into the input neurons
	for(unsigned i = 0; i < inputVals.size(); ++i){
		_layers[0]->getNeuron(i)->setOutput(inputVals[i]);
	}

	// Forward propagate
	for(unsigned layerNum = 1; layerNum < _layers.size(); ++layerNum){
		_layers[layerNum]->updateOutput();
	}
}

void Net::calcError(const std::vector<double> &label)
{
	// Calculate overal net error (RMS of output neuron errors)
	Layer* outputLayer = _layers.back();
	_error = 0.0;

	for(unsigned n = 0; n < outputLayer->getNeuronNum() - 1; ++n)
	{
		double delta = label[n] - outputLayer->getNeuron(n)->getOutput();
		_error += delta *delta;
	}
	_error /= outputLayer->getNeuronNum() - 1; // get average error squared
	_error = sqrt(_error); // RMS

	_recentAverageError = 
			(_recentAverageError * _recentAverageSmoothingFactor + _error)
			/ (_recentAverageSmoothingFactor + 1.0);
}

void Net::calcGradients(const std::vector<double> &label)
{
	for(unsigned layerNum = _layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer* Layer = _layers[layerNum];
		if(layerNum == _layers.size() - 1) 
		{
			Layer->calcGradient(label, true);
		}
		else 
		{
			Layer->calcGradient(label, false);
		}
	}
}

void Net::updateWeights()
{
	// For all layers from outputs to first hidden layer, update edge weights
	for(unsigned layerNum = _layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer* layer = _layers[layerNum];
		layer->updateWeights();
	}

}

void Net::backProp(const std::vector<double> &label)
{
	calcError(label);
	calcGradients(label);
	updateWeights();
}

void Net::getResults(vector<double> &output)
{
	output.clear();
	Layer* outputLayer = _layers.back();

	for(unsigned n = 0; n < outputLayer->getNeuronNum() - 1; ++n)
	{
		output.push_back(outputLayer->getNeuron(n)->getOutput());
	}
}

double Net::getError() const{
	return _recentAverageError;
}