#include "simulation.h"
#include "utils/utils.h"
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <iostream>
#include "utils/engine.h"

void drawCircle(float cx, float cy, float r, int num_segments, bool solid = false) 
{
    if (solid)
        glBegin(GL_TRIANGLE_FAN);
    else
        glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++)   {
        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 
        float x = r * cosf(theta);//calculate the x component 
        float y = r * sinf(theta);//calculate the y component 
        glVertex2f(x + cx, y + cy);//output vertex 
    }
    glEnd();
}

Simulation::Simulation(int iterations, int width, int height, int simulations)
{
    this->iterations = iterations;
    this->iteration = 0;
    this->width = width;
    this->height = height;
    this->simulations = simulations;
    this->simulation = 0;
}

Simulation::~Simulation()
{
    iterations = 0;
    individuals.clear();
}

bool Simulation::run(bool step)
{
    if(step)
    {
        if(iteration < iterations-1)
        {
            update();
            iteration++;
            
            return true;
        }
        else if(simulation < simulations-1)
        {
            simulation++;
            init(individuals);
            
            return true;
        }
        
        simulation = 0;
        
        return false;
    }
    else
    {
        
        for(simulation = 0;simulation < simulations; simulation++)
        {
            init(individuals);
            for(iteration = 0;iteration<iterations;iteration++)
            {
                update();
            }
        }
        
        simulation = 0;
        
        return false;
    }
}

FollowSimulation::FollowSimulation(int iterations, int width, int height, int simulations) : Simulation(iterations, width, height, simulations)
{
    std::cout<<"Creating follow simulation"<<std::endl;
    
    waypoints.resize(20);
    
    for(int i = 0; i<waypoints.size();i++)
    {
        int x = 50 + width/waypoints.size();
        waypoints[i] = Point2d(x, randomNumber(0.0, height));
    }
}
FollowSimulation::~FollowSimulation()
{
}

void FollowSimulation::init(const std::vector<Individual*> individuals)
{
    this->individuals = individuals;
    iteration = 0;
    for(int i = 0;i<this->individuals.size();i++)
    {
        FollowIndividual* individual = (FollowIndividual*)individuals[i];
        individual->currentWaypoint = 0;
        individual->position = Point2d(0.0, height/2);
    }
    
    waypoints.resize(20);
    
    for(int i = 0; i<waypoints.size();i++)
    {
        int x = 50 + i*width/waypoints.size();
        waypoints[i] = Point2d(x, randomNumber(0.0, height));
    }
}
//PARALEL

void FollowSimulation::update()
{
    #pragma omp parallel for
    for(int i=0;i<individuals.size();i++)
    {
        FollowIndividual *individual = (FollowIndividual*)individuals[i];
        if(individual->currentWaypoint < waypoints.size())
        {
            updateIndividual(individual);
        }
    }
}


void FollowSimulation::updateIndividual(FollowIndividual* individual)
{
    Point2d target = waypoints[individual->currentWaypoint];
    
    Point2d directionTarget = (Point2d::direction(individual->position, target) + 1.0) / 2.0;
    double distanceTarget = individual->position.distance(target);
    
    double maxDistance = Point2d(0,0).distance(Point2d(width, height));
    
    if(distanceTarget <= 10) 
    {
        individual->fitness++;
        individual->currentWaypoint++;
        if(individual->currentWaypoint >= waypoints.size())
        {
            return;
        }
        
        target = waypoints[individual->currentWaypoint];
        directionTarget = Point2d::direction(individual->position, target);
        distanceTarget = individual->position.distance(target);
    }
    Point2d nextTarget = individual->currentWaypoint < waypoints.size() - 1 ? waypoints[individual->currentWaypoint + 1] : target;
    
    Point2d normalizedInertia = (individual->inertia.normalized() + 1.0) / 2.0;

    individual->mlp->setInputs(std::vector<double>
    {
        individual->position.x / width,
        individual->position.y / height,
        normalizedInertia.x,
        normalizedInertia.y,
        target.x / width,
        target.y / height,
    });
    std::vector<double> outputs = individual->mlp->getOutputs();
    
    Point2d direction = Point2d(outputs[0], outputs[1]).normalize();
    double acceleration = (outputs[2]/2.0 + 0.5) * individual->acceleration;
    
    individual->inertia += direction * acceleration;
    if (individual->inertia.length() > individual->speed)
        individual->inertia = individual->inertia.normalized() * individual->speed;
    individual->position += individual->inertia;

    individual->position.x = std::max(0.0, individual->position.x);
    individual->position.x = std::min(individual->position.x, (double)width);
    individual->position.y = std::max(0.0, individual->position.y);
    individual->position.y = std::min(individual->position.y, (double)height);
}

void FollowSimulation::display()
{
    for(int i=0;i < waypoints.size();i++)
    {
        glColor3f(1,0,0);
        drawCircle(waypoints[i].x, waypoints[i].y, 10, 32, true);
    }
    glLineWidth(2);
    glBegin(GL_LINES);
        for(int i = 0; i < individuals.size();i++)
        {
            FollowIndividual* individual = (FollowIndividual*)individuals[i];
            std::vector<double> inputs = individual->mlp->getInputs();
            inputs[2] = (inputs[2] - 0.5) * 2.0;
            inputs[3] = (inputs[3] - 0.5) * 2.0;

            glColor3f(1,0,0);
            glVertex2f(individual->position.x, individual->position.y);
            glVertex2f(individual->position.x+inputs[2]*10, individual->position.y + inputs[3]*10);
        }
    glEnd();
    for(int i = 0; i < individuals.size();i++)
    {
        FollowIndividual* individual = (FollowIndividual*)individuals[i];
        glColor3f(0, 1, 0);
        drawCircle(individual->position.x, individual->position.y, 2, 32, true);
    }
    
    glBegin(GL_LINES);
        glColor3f(1,0,0);
        for(int i = 1;i<waypoints.size();i++)
        {
            glVertex2f(waypoints[i-1].x, waypoints[i-1].y);
            glVertex2f(waypoints[i].x, waypoints[i].y);
        }
    glEnd();
}
