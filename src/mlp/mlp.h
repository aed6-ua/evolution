#ifndef __MLP_H__
#define __MLP_H__

#include "layer.h"
#include "neuron.h"
#include <vector>
#include <functional>

class Mlp;
class MlpBuilder;
class MlpDisplay;

class Mlp
{
public:
    ~Mlp();
    
    void setInputs(const std::vector<double> values);
    std::vector<double> getOutputs();
    std::vector<double> getInputs();
    void setWeights(const std::vector<double> &weights);
    void setConnections(const std::vector<bool> &connections);
    std::vector<double> getWeights() const;
    std::vector<bool> getConnections() const;
    int getNumWeights() const;
    
private:
    Mlp();
    
    std::vector<Layer*> layers;
    Layer* inputLayer;
    Layer* outputLayer;
    
    friend class MlpBuilder;
    friend class MlpDisplay;
};

class MlpBuilder
{
public:
    MlpBuilder* withActivation(std::function<double(double)> activation);
    MlpBuilder* withLayer(int n);
    Mlp* build();
    
    static MlpBuilder* withInputs(int n);
private:
    MlpBuilder(int n);    
    
    std::vector<Layer*> layers;
    Layer* inputLayer;
    Layer* outputLayer;
};

class MlpDisplay
{
public:
    static void display(Mlp* mlp, int width, int height);
};

#endif
