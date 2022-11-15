#include "individual.h"
#include "utils/utils.h"
#include "utils/point2d.h"
#include <vector>
#include <omp.h>
#include <math.h>
#include <iostream>

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
Individual::Individual()
{
    this->mlp = NULL;
    this->position = Point2d();
    this->inertia = Point2d(0.0, 0.0);
    this->speed = 0.0;
    this->acceleration = 0.0;
    this->fitness = 0;
};

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
/// <param name="chromosome">The chromosome for the individual.</param>
Individual::Individual(Mlp* mlp, const Point2d &position) : Individual()
{ 
	this->mlp = mlp;
	this->position = position;
};

/// <summary>
/// Destroys this instance of Individual.
/// </summary>
Individual::~Individual()
{
    if(mlp != NULL)
        delete mlp;
    mlp = NULL;
}

/// <summary>
/// Creates a new Individual resulting of the combination of this instance and another Individual.
/// </summary>
/// <param name="parent2">The other Individual to create the combination.</param>
void Individual::mate(const Individual &par2, Individual *child) 
{ 
    const double p1Chance = 0.49;
    const double p2Chance = 0.98;
	// chromosome for offspring
	std::vector<double> childWeights = mlp->getWeights();
	std::vector<double> parentWeights = par2.mlp->getWeights();
	
	int cut = randomNumber(0.0, 1.0) * childWeights.size();
	
	for(int i=0;i<cut;i++)
	{
	    double p = randomNumber(0.0, 1.0);
	    if(p > 0.98)
	        childWeights[i] = randomNumber(-1.0, 1.0);
	}
	
	for(int i=cut; i<childWeights.size();i++)
	{
	    double p = randomNumber(0.0, 1.0);
	    if(p > 0.98)
	        childWeights[i] = randomNumber(-1.0, 1.0);
	    else
	        childWeights[i] = parentWeights[i];
	}
    
    child->mlp->setWeights(childWeights);
    
    std::vector<bool> childConnections = mlp->getConnections();
	std::vector<bool> parentConnections = par2.mlp->getConnections();
	
	for(int i=0;i<cut;i++)
	{
	    double p = randomNumber(0.0, 1.0);
	    if(p > 0.98)
	        childConnections[i] = randomNumber(0.0, 1.0) >= 0.5;
	}
	
	for(int i=cut; i<childWeights.size();i++)
	{
	    double p = randomNumber(0.0, 1.0);
	    if(p > 0.98)
	        childConnections[i] = randomNumber(0.0, 1.0) >= 0.5;
	    else
	        childConnections[i] = parentConnections[i];
	}
    
    child->mlp->setConnections(childConnections);
};

