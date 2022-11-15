#include "layer.h"
#include "activation.h"
#include <cassert>

Layer::Layer(const std::vector<Neuron*> &neurons)
{
    this->activation = activation::linear;
    
    this->neurons.resize(neurons.size());
    for(int i = 0; i < neurons.size(); i++)
    {
        this->neurons[i] = neurons[i];
    }
}

Layer::Layer(const std::vector<Neuron*> &neurons, std::function<double(double)> activation) : Layer(neurons)
{
    this->activation = activation;
}
Layer::~Layer()
{
    for(int i = 0; i < neurons.size(); i++)
    {
        delete neurons[i];
    }
    
    neurons.clear();
}

std::vector<double> Layer::getOutput()
{
    std::vector<double> outputs(neurons.size());
    for(int i = 0; i < neurons.size(); i++)
    {
        outputs[i] = this->neurons[i]->updateOutput(this->activation);
    }
    
    return outputs;
}
