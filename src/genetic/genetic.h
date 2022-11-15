#ifndef __GENETIC_H__
#define __GENETIC_H__

#include <vector>
#include "individual.h"
#include "simulation.h"
#include <functional>
#include <string>

class Genetic
{
public:
    Genetic(int population, const std::string &fileName, std::function<Individual*()> createRandomIndividual, Simulation* simulation);
    ~Genetic();
    virtual void initialize();
    virtual bool load();
    virtual void save();
    virtual void updateAndEvolve();

    int population;
    std::vector<Individual*> individuals;
    int generation;
    std::string fileName;
    
    std::function<Individual*()> createRandomIndividual;
    
    Simulation *simulation;
    
protected:
    virtual std::vector<Individual*> bestIndividuals();
    std::vector<Individual*> nextGeneration();
    double simulationStartTime;
    
    
};

#endif
