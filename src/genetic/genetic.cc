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

bool Genetic::load()
{
    if(true)//fileName.empty())
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

        // Crea un bloque paralelo utilizando OpenMP
        #pragma omp parallel for
        for (int i = 0; i < individuals.size(); i++)
        {
            // Obtiene los pesos y conexiones del individuo actual
            std::vector<double> weights = individuals[i]->mlp->getWeights();
            std::vector<bool> connections = individuals[i]->mlp->getConnections();

            // Escribe los pesos en el archivo de manera concurrente
            of.write(reinterpret_cast<char*>(&weights[0]), sizeof(double)*weights.size());

            // Convierte los valores de conexión a un vector de tipo char
            // para poder escribirlos en el archivo de manera más sencilla
            std::vector<char> temp(connections.size());
            for (int j = 0; j < connections.size(); j++)
            {
                temp[j] = connections[j] ? 1 : 0;
            }

            // Escribe las conexiones en el archivo de manera concurrente
            of.write(reinterpret_cast<char*>(&temp[0]), sizeof(char)*temp.size());
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

