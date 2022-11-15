#ifndef __INDIVIDUAL_H__
#define __INDIVIDUAL_H__

#include "mlp/mlp.h"
#include "utils/point2d.h"
#include <functional>

/// <summary>
/// A class that represents an individual member of the population.
/// </summary>
class Individual 
{ 
public: 
    Mlp* mlp;
    
    double speed;
    double acceleration;
    
    Point2d position;
    Point2d inertia;

    /// <summary>
    /// The fitness of this individual.
    /// </summary>
	int fitness; 
    
    /// <summary>
    /// Initializes a new instance of Individual.
    /// </summary>
    Individual();
    
    /// <summary>
    /// Initializes a new instance of Individual.
    /// </summary>
    /// <param name="chromosome">The chromosome for the individual.</param>
	Individual(Mlp* mlp, const Point2d &position); 

    /// <summary>
    /// Destroys this instance of Individual.
    /// </summary>
    ~Individual();

    /// <summary>
    /// Creates a new Individual resulting of the combination of this instance and another Individual.
    /// </summary>
    /// <param name="parent2">The other Individual to create the combination.</param>
	virtual void mate(const Individual &parent2, Individual *child); 

    /// <summary>
    /// Calculates the fitness value of this Individual's chromosome.
    /// </summary>
	virtual int calculateFitness() = 0;
};

class FollowIndividual : public Individual
{
public:
    int currentWaypoint;
    
	FollowIndividual(Mlp* mlp, const Point2d &position) : Individual(mlp, position)
	{
	    currentWaypoint = 0;
	    speed = 4;
	    acceleration = 0.5;
	}
    
    virtual int calculateFitness() override { return fitness; };
};

#endif
