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

#include <time.h>

#include <mpi.h>

#include <sstream>


using namespace std;

int POB=500;
int MESSAGE_SIZE=285508;
int NUM_THREADS=1;
std::vector<double> execution_times;
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
                POB,
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
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double sstart = omp_get_wtime();
    double start = omp_get_wtime();
    if(!step && genetic->generation > 0 && genetic->generation % 50 != 0) {
        char* oldGen;
        while(genetic->generation % 50 != 0) {
            if (genetic->generation % 50 == (50 - 1))
                oldGen = genetic->updateAndEvolveLast();
            else
                genetic->updateAndEvolve();
        }
        // Sincronización de los procesos
        char array[MESSAGE_SIZE];
        char* newGen;
        newGen = (char*)malloc(MESSAGE_SIZE);
        if (world_rank == 0) {
            std::vector<std::vector<Individual*>> generations;
            generations.push_back(genetic->Genetic::deserializeGeneration(oldGen));
            for (int i = 1; i < world_size; i++) {
                MPI_Recv(array, MESSAGE_SIZE, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                generations.push_back(genetic->Genetic::deserializeGeneration(array));
            }
            std::vector<Individual*> generation = genetic->Genetic::combineGenerations(generations);
            newGen = genetic->Genetic::serializeGeneration(generation);
        }
        else {
            MPI_Send(oldGen, MESSAGE_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(newGen, MESSAGE_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
        genetic->individuals = genetic->Genetic::deserializeGeneration(newGen);
        free(newGen);
        if (world_rank!=0) {
            printf("Generación actualizada con la generación combinada\n");
        }
    }
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

    //Hacer solo 100 iteraciones
    /*
    if (genetic->generation > 99){
        std::ofstream outputFile;
        outputFile.open("execution_times.txt", std::ios_base::app);
        outputFile << POB << " " << NUM_THREADS << " " << omp_get_wtime() - sstart << std::endl;
        outputFile.close();
        glutLeaveMainLoop();
    }*/
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

    //Hacer solo 100 iteraciones
    //if (genetic->generation > 99)
    //    glutLeaveMainLoop();
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
        simulationName = argv[1];
        //POB = atoi(argv[2]);
        //NUM_THREADS = atoi(argv[3]);
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
    // Inicializar MPI
    MPI_Init(NULL, NULL);

    if(!processArgs(argc, argv, simulationName, step))
    {
        return 0;
    }
    //std::cout<<"Población: "<<POB<<std::endl;
    //omp_set_num_threads(NUM_THREADS);
    //std::cout<<"Threads: "<<NUM_THREADS<<std::endl;

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

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}
