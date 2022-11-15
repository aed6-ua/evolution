#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#include <vector>
#include "individual.h"
#include "utils/point2d.h"

class Simulation
{
public:
    Simulation(int iterations, int width, int height, int simulations = 1);
    virtual ~Simulation();
    
    virtual void init(const std::vector<Individual*> individuals) = 0;
    bool run(bool step);
    virtual void display() = 0;
    
    int simulations;
    int iterations;
    int width;
    int height;
    std::vector<Individual*> individuals;
    
protected:
    virtual void update() = 0;
    
    int iteration;
    int simulation;
};

class FollowSimulation : public Simulation
{
public:
    FollowSimulation(int iterations, int width, int height, int simulations=1);
    virtual ~FollowSimulation();
    
    virtual void init(const std::vector<Individual*> individuals) override;
    virtual void display() override;
    
    std::vector<Point2d> waypoints;
    
protected:
    virtual void update() override;
    virtual void updateIndividual(FollowIndividual* individual);
};

#endif
