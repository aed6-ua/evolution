#ifndef __NEURON_H__
#define __NEURON_H__

#include <vector>
#include "activation.h"
#include <functional>

class Neuron
{
public:
    Neuron();
    Neuron(const std::vector<Neuron*> &inputs, const std::vector<double> &weights, double bias = 0.0);
    Neuron(const std::vector<Neuron*> &inputs, double bias = 0.0);
    Neuron(const Neuron &neuron);
    ~Neuron();
    
    void setBias(double bias) { this->bias = bias; };
    void setWeights(const std::vector<double> &weights);
    void setConnections(const std::vector<bool> &connections);
    
    std::vector<double> getWeights() const { return weights; }
    std::vector<bool> getConnections() const { return connections; }
    virtual double getOutput() const { return output; };
    double getNumInputs() const { return inputs.size(); };
    
    virtual double updateOutput(std::function<double(double)> activation);
    
protected:
    std::vector<double> weights;
    std::vector<bool> connections;
    std::vector<Neuron*> inputs;
    double output;
    double bias;
};

class InputNeuron : public Neuron
{
public: 
    InputNeuron();
    
    void setInputValue(double value) { inputValue = value; };
    double getInputValue() const { return inputValue; };
    virtual double updateOutput(std::function<double(double)> activation = activation::linear) override;
private:
    double inputValue;
};

#endif
