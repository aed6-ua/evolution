#include "neuron.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>

Neuron::Neuron()
{
    bias = 0.0;
    output = 0.0;
}
Neuron::Neuron(const std::vector<Neuron*> &inputs, const std::vector<double> &weights, double bias)
{
    assert(inputs.size() == weights.size());
    
    this->bias = bias;
    this->inputs.clear();
    this->inputs.resize(inputs.size());
    this->weights.clear();
    this->weights.resize(weights.size());
    this->connections.clear();
    this->connections.resize(inputs.size());
    
    for(int i = 0; i < inputs.size(); i++)
    {
        this->inputs[i] = inputs[i];
        this->weights[i] = weights[i];
        this->connections[i] = true;
    }
}
Neuron::Neuron(const std::vector<Neuron*> &inputs, double bias)
{
    this->bias = bias;
    this->inputs.clear();
    this->inputs.resize(inputs.size());
    this->weights.clear();
    this->weights.resize(inputs.size());
    this->connections.clear();
    this->connections.resize(inputs.size());
    
    for(int i = 0; i < inputs.size(); i++)
    {
        this->inputs[i] = inputs[i];
        this->weights[i] = 0.5;
        this->connections[i] = true;
    }
}
Neuron::Neuron(const Neuron &neuron) : Neuron(neuron.inputs, neuron.weights, neuron.bias)
{
}
Neuron::~Neuron()
{
    this->bias = 0.0;
    this->inputs.clear();
    this->weights.clear();
    this->connections.clear();
    this->output = 0.0;
}

double Neuron::updateOutput(std::function<double(double)> activation)
{
    std::vector<double> inputValues(inputs.size());
    
    for(int i = 0; i < inputs.size(); i++)
    {
        inputValues[i] = connections[i] ? inputs[i]->getOutput() : 0.0;
    }
    
    double dot = 0.0;
    for(int i=0;i<inputValues.size();i++)
    {
        dot+=inputValues[i]*weights[i];
    }
        
    output = activation(dot) + bias;
    
    return output;
}

void Neuron::setWeights(const std::vector<double> &weights)
{
    assert(weights.size() == this->weights.size());
    
    for(int i = 0; i < weights.size(); i++)
    {
        this->weights[i] = weights[i];
    }
}

void Neuron::setConnections(const std::vector<bool> &connections)
{
    assert(connections.size() == this->connections.size());
    
    for(int i = 0; i < connections.size(); i++)
    {
        this->connections[i] = connections[i];
    }
}

InputNeuron::InputNeuron()
{
    inputValue = 0.0;
    output = 0.0;
}
    
double InputNeuron::updateOutput(std::function<double(double)> activation)
{
    output = activation(inputValue);
    
    return output;
}
