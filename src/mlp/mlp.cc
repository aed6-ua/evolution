#include "mlp.h"
#include "layer.h"
#include "neuron.h"

#include <cassert>

#include "utils/engine.h"

#include <iostream>

Mlp::Mlp()
{
    inputLayer = NULL;
    outputLayer = NULL;
}
Mlp::~Mlp()
{
    for(int i = 0; i < layers.size(); i++)
        delete layers[i];
        
    inputLayer = NULL;
    outputLayer = NULL;
    layers.clear();
}

void Mlp::setInputs(const std::vector<double> values)
{
    std::vector<Neuron*> neurons = inputLayer->getNeurons();
    assert(values.size() == neurons.size());
    
    for(int i = 0; i < values.size(); i++)
    {
        InputNeuron* input = (InputNeuron*)neurons[i];
        input->setInputValue(values[i]);
    }
}

std::vector<double> Mlp::getOutputs()
{
    for(int i = 0; i < layers.size() - 1; i++)
    {
        layers[i]->getOutput();
    }
    
    return outputLayer->getOutput();
}

std::vector<double> Mlp::getInputs()
{
    return inputLayer->getOutput();
}

void Mlp::setWeights(const std::vector<double> &weights)
{
    int numWeights = getNumWeights();
        
    assert(numWeights == weights.size());
    
    std::vector<double>::const_iterator it = weights.begin();
    
    for(int i = 1; i < layers.size(); i++)
    {
        int previousLayerNeurons = layers[i - 1]->getNeurons().size();
        std::vector<Neuron*> neurons = layers[i]->getNeurons();
        int currentLayerNeurons = neurons.size();
        
        for(int j = 0; j < neurons.size(); j++)
        {
            std::vector<double> subWeights(it, it + previousLayerNeurons);
            neurons[j]->setWeights(subWeights);
        
            it = it + previousLayerNeurons;
        }
    }
}

std::vector<double> Mlp::getWeights() const
{
    std::vector<double> weights(getNumWeights());
    
    int it = 0;
    
    for(int i = 1; i < layers.size(); i++)
    {
        int previousLayerNeurons = layers[i - 1]->getNeurons().size();
        std::vector<Neuron*> neurons = layers[i]->getNeurons();
        int currentLayerNeurons = neurons.size();
        
        for(int j = 0; j < neurons.size(); j++)
        {
            std::vector<double> subWeights = neurons[j]->getWeights();
            
            for(int k = 0; k < subWeights.size(); k++)
            {
                weights[it] = subWeights[k];
                it++;
            }
        }
    }
    
    return weights;
}

void Mlp::setConnections(const std::vector<bool> &connections)
{
    int numWeights = getNumWeights();
        
    assert(numWeights == connections.size());
    
    std::vector<bool>::const_iterator it = connections.begin();
    
    for(int i = 1; i < layers.size(); i++)
    {
        int previousLayerNeurons = layers[i - 1]->getNeurons().size();
        std::vector<Neuron*> neurons = layers[i]->getNeurons();
        int currentLayerNeurons = neurons.size();
        
        for(int j = 0; j < neurons.size(); j++)
        {
            std::vector<bool> subConnections(it, it + previousLayerNeurons);
            neurons[j]->setConnections(subConnections);
        
            it = it + previousLayerNeurons;
        }
    }
}

std::vector<bool> Mlp::getConnections() const
{
    std::vector<bool> connections(getNumWeights());
    
    int it = 0;
    
    for(int i = 1; i < layers.size(); i++)
    {
        int previousLayerNeurons = layers[i - 1]->getNeurons().size();
        std::vector<Neuron*> neurons = layers[i]->getNeurons();
        int currentLayerNeurons = neurons.size();
        
        for(int j = 0; j < neurons.size(); j++)
        {
            std::vector<bool> subConnections = neurons[j]->getConnections();
            
            for(int k = 0; k < subConnections.size(); k++)
            {
                connections[it] = subConnections[k];
                it++;
            }
        }
    }
    
    return connections;
}

int Mlp::getNumWeights() const
{
    int numWeights = 0;
    for(int i = 1; i < layers.size(); i++)
        numWeights += layers[i]->getNeurons().size() * layers[i - 1]->getNeurons().size();
        
    return numWeights;
}

MlpBuilder* MlpBuilder::withInputs(int n)
{
    return new MlpBuilder(n);
}

MlpBuilder::MlpBuilder(int n)
{
    std::vector<Neuron*> neurons(n);
    
    for(int i = 0; i < n; i++)
    {
        neurons[i] = new InputNeuron();
    }
    
    inputLayer = new Layer(neurons);
    outputLayer = inputLayer;
    
    layers.push_back(inputLayer);
}

MlpBuilder* MlpBuilder::withActivation(std::function<double(double)> activation)
{
    layers[layers.size()-1]->setActivation(activation);
    
    return this;
}

MlpBuilder* MlpBuilder::withLayer(int n)
{
    std::vector<Neuron*> inputs = layers[layers.size()-1]->getNeurons();
    std::vector<Neuron*> neurons(n);
    
    for(int i = 0; i < n; i++)
    {
        neurons[i] = new Neuron(inputs);
    }
    
    outputLayer = new Layer(neurons);
    layers.push_back(outputLayer);
    
    return this;
}

Mlp* MlpBuilder::build()
{
    assert(inputLayer != NULL);
    assert(outputLayer != NULL);
    assert(layers.size() >= 3);
    
    Mlp* mlp = new Mlp();
    
    mlp->layers = layers;
    mlp->inputLayer = inputLayer;
    mlp->outputLayer = outputLayer;
    
    return mlp;
}

void MlpDisplay::display(Mlp* mlp, int width, int height)
{
    Engine engine;
    double xOffset = width/(mlp->layers.size() + 1);
    
    std::vector<Neuron*> previousNeurons = mlp->inputLayer->getNeurons();
    std::vector<Point2d> previousPositions(previousNeurons.size());
    double yOffset = height / (previousNeurons.size()+1);
    
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
    for(int i=0;i<previousNeurons.size();i++)
    {
        r = previousNeurons[i]->getOutput() > 0.0 ? std::abs(previousNeurons[i]->getOutput()) : 0.0;
        b = previousNeurons[i]->getOutput() < 0.0 ? std::abs(previousNeurons[i]->getOutput()) : 0.0;
        previousPositions[i] = Point2d(xOffset, yOffset*(i+1));
        engine.drawCircle(previousPositions[i], 10, r,g,b,true);
    }
    
    for(int i=1;i<mlp->layers.size();i++)
    {
        std::vector<Neuron*> neurons = mlp->layers[i]->getNeurons();
        std::vector<Point2d> positions(neurons.size());
        
        // Draw neurons
        for(int j=0;j<neurons.size();j++)
        {
            yOffset = height / (neurons.size()+1);
            positions[j] = Point2d(xOffset*(i+1), yOffset*(j+1));
            r = neurons[j]->getOutput() > 0.0 ? std::abs(neurons[j]->getOutput()) : 0.0;
            b = neurons[j]->getOutput() < 0.0 ? std::abs(neurons[j]->getOutput()) : 0.0;
            engine.drawCircle(positions[j], 10, r,g,b,true);
            
            // Draw connections
            std::vector<bool> connections = neurons[j]->getConnections();
            std::vector<double> weights = neurons[j]->getWeights();
            for(int k = 0;k<connections.size();k++)
            {
                if(connections[k])
                {
                    //std::cout<<"write line"<<positions[j].x<<" "<<positions[j].y<<" -> "<<previousPositions[k].x<<" "<<previousPositions[k].y<<std::endl;
                    r = weights[k] > 0.0 ? std::abs(weights[k]) : 0.0;
                    b = weights[k] < 0.0 ? std::abs(weights[k]) : 0.0;
                    engine.drawLine(positions[j], previousPositions[k], r,g,b);
                }
            }
        }
        
        previousPositions = positions;
        previousNeurons = neurons;
    }
}
