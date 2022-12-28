#include <vector>
#include <algorithm>
#include "genetic.h"
#include "individual.h"
#include "mlp/mlp.h"
#include "utils/utils.h"
#include <thread>
#include <random>
#include <fstream>
#include <iostream>
#include <functional>
#include <omp.h>
#include <string.h>
#include <sstream>

using namespace std;

Genetic::Genetic(int population, const std::string &fileName, std::function<Individual*()> createRandomIndividual, Simulation* simulation)
{
    this->population = population;
    this->createRandomIndividual = createRandomIndividual;
    this->simulation = simulation;
    this->fileName = fileName;
    this->simulationStartTime = 0.0;
    this->generation = 1;
}

Genetic::~Genetic()
{
    for(int i = 0; i < individuals.size();i++)
    {
        delete individuals[i];
    }
    
    individuals.clear();
    
    if(simulation != NULL)
    {
        delete simulation;
        simulation = NULL;
    }
    
    std::cout<<"Genetic destroyed"<<std::endl;
}

void Genetic::initialize()
{
    if(!load())
    {
        std::cout<<"Initializing..."<<std::endl;
        individuals.clear();
        individuals.resize(population);
        
        #pragma omp parallel for
        for (int i = 0; i < population; i++)
        {
            // Inicializa cada individuo de la población de manera concurrente
            individuals[i] = createRandomIndividual();
        }
    }
    this->simulationStartTime = omp_get_wtime();
    simulation->init(individuals);
}

char* Genetic::serializeGeneration(std::vector<Individual*> individualsLocal)
{
    std::stringstream ss;
    int individualsSize = individualsLocal.size();
    ss.write(reinterpret_cast<char*>(&generation), sizeof(generation));
    ss.write(reinterpret_cast<char*>(&individualsSize), sizeof(individualsSize));
    for(int i =0;i<individualsLocal.size();i++)
    {
        std::vector<double> weights = individualsLocal[i]->mlp->getWeights();
        ss.write(reinterpret_cast<char*>(&weights[0]), sizeof(double)*weights.size());
        std::vector<bool> connections = individualsLocal[i]->mlp->getConnections();
        //ss.write(reinterpret_cast<char*>(&connections), sizeof(bool)*connections.size());
        std::copy(connections.begin(), connections.end(), std::ostreambuf_iterator<char>(ss));
        for(int i=0;i<individualsLocal.size();i++)
        {
            individualsLocal[i]->calculateFitness();
        }
        ss.write(reinterpret_cast<char*>(&(individualsLocal[i]->fitness)), sizeof(int));
    }
    std::string str = ss.str();
    char* buffer = new char[str.size()];
    memcpy(buffer, str.c_str(), str.size());
    return buffer;
} 

std::vector<Individual*> Genetic::deserializeGeneration(char* buffer)
{
    std::stringstream ss;
    ss.write(buffer, sizeof(buffer));
    int generationLocal;
    int populationLocal;
    ss.read(reinterpret_cast<char*>(&generationLocal), sizeof(generationLocal));
    ss.read(reinterpret_cast<char*>(&populationLocal), sizeof(populationLocal));
    std::vector<Individual*> individualsLocal;
    individualsLocal.clear();
    individualsLocal.resize(populationLocal);
    for(int i =0;i<individualsLocal.size();i++)
    {
        individualsLocal[i] = createRandomIndividual();
        Mlp* mlp = individualsLocal[i]->mlp;
        int numWeights = mlp->getNumWeights();
        std::vector<double> weights(numWeights);
        
        ss.read(reinterpret_cast<char*>(&weights[0]), sizeof(double)*weights.size());
        mlp->setWeights(weights);

        std::vector<bool> connections(numWeights);
        std::vector<char> temp(numWeights);
        ss.read(reinterpret_cast<char*>(&temp[0]), sizeof(char)*connections.size());
        for(int j=0;j<numWeights;j++)
        {
            connections[j] = temp[j] == 1;
        }
        mlp->setConnections(connections);
        int fitness;
        ss.read(reinterpret_cast<char*>(&fitness), sizeof(int));
        individualsLocal[i]->fitness = fitness;
    }
    return individualsLocal;
}

std::vector<Individual*> Genetic::combineGenerations(std::vector<std::vector<Individual*>> &generations)
{
    vector<Individual*> combined;
    // Combinar las generaciones y quedarse con los 500 mejores
    if (!generations.empty()) {
        for(auto const &v: generations)
        {
            combined.insert(combined.end(), v.begin(), v.end());
        }
        std::sort(combined.begin(), combined.end(), [](Individual *a, Individual *b)
        {
            return a->fitness > b->fitness;
        });
        combined.resize(500);
    }
    return combined;
}

bool Genetic::load()
{
    if(fileName.empty())
    {
        std::cout<<"No generation file specified"<<std::endl;
        
        return false;
    }
    
    std::cout<<"Loading generation file "<<fileName<<std::endl;
        
    std::ifstream ifs(fileName.c_str(), std::ios::binary);
    if(ifs.is_open())
    {
        std::cout<<"Reading previous state..."<<std::endl;
        ifs.read(reinterpret_cast<char*>(&generation), sizeof(generation));
        std::cout<<"Generation "<<generation<<std::endl;
        ifs.read(reinterpret_cast<char*>(&population), sizeof(population));
        std::cout<<"Individuals "<<population<<std::endl;
        individuals.clear();
        individuals.resize(population);
        for(int i =0;i<individuals.size();i++)
        {
            individuals[i] = createRandomIndividual();
            Mlp* mlp = individuals[i]->mlp;
            int numWeights = mlp->getNumWeights();
            std::vector<double> weights(numWeights);
            
            ifs.read(reinterpret_cast<char*>(&weights[0]), sizeof(double)*weights.size());
            mlp->setWeights(weights);
            
            std::vector<bool> connections(numWeights);
            std::vector<char> temp(numWeights);
            ifs.read(reinterpret_cast<char*>(&temp[0]), sizeof(char)*connections.size());
            for(int j=0;j<numWeights;j++)
            {
                connections[j] = temp[j] > 0;
            }
            
            mlp->setConnections(connections);
        }
        ifs.close();
        
        return true;
    }
    
    std::cout<<"Could not open generation file "<<fileName<<std::endl;
    
    return false;
}

void Genetic::save()
{
    if(fileName.empty()) 
        return;
        
    std::ofstream of(fileName.c_str(), std::ios::binary);
    if(of.is_open())
    {
        int individualsSize = individuals.size();
        of.write(reinterpret_cast<char*>(&generation), sizeof generation);
        of.write(reinterpret_cast<char*>(&individualsSize), sizeof individualsSize);
        for(int i = 0;i<individuals.size();i++)
        {
            std::vector<double> weights = individuals[i]->mlp->getWeights();
            of.write(reinterpret_cast<char*>(&weights[0]), sizeof(double)*weights.size());
            std::vector<bool> connections = individuals[i]->mlp->getConnections();
            //of.write(reinterpret_cast<char*>(&connections), sizeof(bool)*connections.size());
            std::copy(connections.begin(), connections.end(), std::ostreambuf_iterator<char>(of));
        }
        of.close();
    }
}


void Genetic::updateAndEvolve()
{
    if(!simulation->run(true))
    {
        if (this->simulationStartTime > 0.0)
        {
            double totalSimulationTime = omp_get_wtime() - this->simulationStartTime;
            std::cout<<"Total simulation time: "<<totalSimulationTime<<std::endl;
            /*
            std::ofstream outputFile;
            outputFile.open("simulation_times.txt", std::ios_base::app);
            outputFile << totalSimulationTime << std::endl;
            outputFile.close();*/
        }
        this->simulationStartTime = omp_get_wtime();
        
        std::vector<Individual*> newGeneration = nextGeneration();
        
        for(int i = 0;i < individuals.size(); i++)
        {
            delete individuals[i];
        }
        individuals = newGeneration;
        generation++;
        std::cout<<"Generation "<<generation<<std::endl;
        save();
        
        simulation->init(individuals);
    }
}

//PARALEL

std::vector<Individual*> Genetic::nextGeneration()
{
    std::vector<Individual*> newGeneration(individuals.size());
    std::vector<Individual*> best = bestIndividuals();
    
    #pragma omp parallel for
    for(int i = 0; i < individuals.size(); i++)
    {
        if (i < best.size()/2)
        {
            // Perform elitism, best individuals pass directly to next generation
            Individual *elite = createRandomIndividual();
            elite->mlp->setWeights(best[i]->mlp->getWeights());
            elite->mlp->setConnections(best[i]->mlp->getConnections());
            newGeneration[i] = elite;
        }
        else
        {
            // The remaining indiviuals are combination of two random individuals from the best
            Individual *child = createRandomIndividual();
            int a = randomNumber(0.0, 1.0) * (best.size()-1);
            int b = randomNumber(0.0, 1.0) * (best.size()-1);
            Individual *parent1 = best[a];
            Individual *parent2 = best[b];
            
            parent1->mate(*parent2, child);
            
            newGeneration[i] = child;  
        }
    }
    
    return newGeneration;
}


std::vector<Individual*> Genetic::bestIndividuals()
{
    //#pragma omp parallel for
    for(int i=0;i<individuals.size();i++)
    {
        individuals[i]->calculateFitness();
    }

    sort(individuals.begin(), individuals.end(), [](Individual *a, Individual *b)
    {
        return a->fitness > b->fitness;
    });
    
    for(int i = 0;i<10;i++)
        std::cout<<individuals[i]->fitness<<"\t";
    std::cout<<std::endl;
    
    return std::vector<Individual*>(individuals.begin(), individuals.begin() + ((int)(individuals.size()*0.2)));
}
