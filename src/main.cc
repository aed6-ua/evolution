#include "mlp/mlp.h"
#include "mlp/activation.h"
#include "genetic/individual.h"
#include "genetic/genetic.h"
#include "utils/utils.h"
#include "utils/point2d.h"
#include <vector>
#include <iostream>
#include <thread>
#include <random>
#include <algorithm>

#include <GL/gl.h>
#include <GL/freeglut.h>
#include <omp.h>

#include <fstream>
#include <string.h>
#include <limits>
#include <map>

int width = 500, height = 500;
bool display = false;
bool step = false;
std::string simulationName = "";
Genetic *genetic = NULL;
std::map<std::string, std::function<Genetic*(const std::string&)>> factory = 
{
   {
        "follow", 
        [](const std::string &fileName)
        {
            return new Genetic(
                500,
                fileName,
                []()
                {
                    Mlp* mlp = MlpBuilder::withInputs(6)
                            ->withLayer(7)
                            ->withActivation(activation::tanH)
                            ->withLayer(3)
                            ->withActivation(activation::tanH)
                            ->build();
                            
                    int elements = mlp->getNumWeights();
                    std::vector<double> y(elements);    
                    for(int j = 0; j<elements;j++)
                    {
                        y[j] = randomNumber(-1.0, 1.0); 
                    }
                    mlp->setWeights(y);
                    
                    std::vector<bool> x(elements);    
                    for(int j = 0; j<elements;j++)
                    {
                        x[j] = randomNumber(0.0, 1.0) >= 0.75; 
                    }
                    mlp->setConnections(x);   
                    
                    return new FollowIndividual(mlp, Point2d(randomNumber(0.0, width), randomNumber(0.0, height)));
                },
                new FollowSimulation(500, width, height, 2));
        }
   }
};

void resizeSimulation(GLsizei w, GLsizei h) 
{
    width = w;
    height = h;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void resizeNetwork(GLsizei w, GLsizei h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, 0, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void displaySimulation(void)
{
    double start = omp_get_wtime();
    if(!step && genetic->generation > 0 && genetic->generation % 100 != 0)
        while(genetic->generation % 100 != 0)
            genetic->updateAndEvolve();
    else
        genetic->updateAndEvolve();
    double updateTime = omp_get_wtime() - start;
    
    start = omp_get_wtime();
    glClearColor(0,0,0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    
    double xScale = width / (double)genetic->simulation->width;
    double yScale = height / (double)genetic->simulation->height;
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glScalef(xScale, yScale, 1);
    genetic->simulation->display();
    glPopMatrix();
    
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    std::cout<<"Update time: "<<updateTime<<" Display time: "<<omp_get_wtime() - start<<std::endl;
}

void displayNetwork(void)
{
    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT);
    static int index = 0;
    
    int xOffset = width/3;
    int yOffset = height/3;
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glScalef(1.0/3.0, 1.0/3.0, 1);
    for(int i=0;i<3;i++)
    {
        glPushMatrix();
        for(int j=0;j<3;j++)
        {
            MlpDisplay::display(genetic->individuals[i*3 + j]->mlp, width, height);
            glTranslatef(0, height, 0);
        }
        glPopMatrix();
        glTranslatef(width, 0, 0);
    }
    glPopMatrix();
    
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

bool processArgs(int argc, char** argv, std::string &simulationName, bool &step)
{
    bool error = false;
    if(argc <= 1)
    {
        std::cout<<"No simulation specified"<<std::endl;
        error = true;
    }
    else if(argc == 2)
    {
        step = false;
        simulationName = argv[1];
    }
    else if(argc == 3)
    {
        if(std::string(argv[1]) == "-s")
        {
            step = true;
            simulationName = argv[2];
        }
        else if(std::string(argv[2]) == "-s")
        {
            step = true;
            simulationName = argv[1];
        }
        else
        {
            error = true;
            std::cout<<"Invalid argument specified: "<<argv[1]<<" "<<argv[2]<<std::endl;
            std::cout<<"Usage: evolution simulationName [-s]"<<std::endl;
            std::cout<<"\t-s:\t If specified, all generations are displayed, otherwise, only the generations multiple of 100 are displayed."<<std::endl;
        }
    }
    
    if(error || factory.find(simulationName) == factory.end())
    {
        if(!error)
            std::cout<<"There is no simulation called "<<simulationName<<std::endl;
        std::cout<<"\tValid simulation names are:"<<std::endl;
        for(std::map<std::string, std::function<Genetic*(const std::string&)>>::iterator it = factory.begin();it != factory.end();it++)
            std::cout<<"\t"<<it->first<<std::endl;
            
        return false;
    }
    
    return true;
}

int main(int argc, char** argv)
{
    if(!processArgs(argc, argv, simulationName, step))
    {
        return 0;
    }
    
    genetic = factory[simulationName](simulationName);
    genetic->initialize();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(argv[1]);
    glutDisplayFunc(displaySimulation);
    glutReshapeFunc(resizeSimulation);
    
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(600, 100);
    glutCreateWindow("network");
    glutDisplayFunc(displayNetwork);
    glutReshapeFunc(resizeNetwork);
    
    glutMainLoop();
    
    return 0;
}
