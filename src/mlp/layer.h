#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>
#include <algorithm>
#include "neuron.h"
#include <functional>

class Layer
{
public:
    Layer(const std::vector<Neuron*> &neurons);
    Layer(const std::vector<Neuron*> &neurons, std::function<double(double)> activation);
    ~Layer();
    
    void setActivation(std::function<double(double)> activation) { this->activation = activation; };
    
    std::vector<double> getOutput();
    const std::vector<Neuron*>& getNeurons() { return neurons; }
private:
    std::vector<Neuron*> neurons;
    std::function<double(double)> activation;
};

#endif
