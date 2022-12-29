#ifndef __GENETIC_H__
#define __GENETIC_H__

#include <vector>
#include "individual.h"
#include "simulation.h"
#include <functional>
#include <string>

using namespace std;

class Genetic
{
public:
    Genetic(int population, const std::string &fileName, std::function<Individual*()> createRandomIndividual, Simulation* simulation);
    ~Genetic();
    virtual void initialize();
    virtual std::vector<Individual*> deserializeGeneration(char*);
    virtual char* serializeGeneration(vector<Individual*>);
    virtual std::vector<Individual*> combineGenerations(std::vector<std::vector<Individual*>>&);
    virtual bool load();
    virtual void save();
    virtual void updateAndEvolve();
    virtual char* updateAndEvolveLast();

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
