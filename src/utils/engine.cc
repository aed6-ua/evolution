#include "engine.h"
#include <cassert>
#include "point2d.h"
#include "math.h"

#include <GL/gl.h>
#include <GL/freeglut.h>

Engine::Engine()
{
}
Engine::~Engine()
{
}

void Engine::drawCircle(const Point2d &center, float radius, float r, float g, float b, bool solid)
{
    glColor3f(r, g, b);
    if(solid)
    {
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(center.x, center.y);
        for (int i = 0; i < 33; i++)   {
            float theta = 2.0f * 3.1415926f * float(i) / 32.0f;
            float x = radius * cosf(theta);
            float y = radius * sinf(theta);
            glVertex2f(x + center.x, y + center.y);
        }
        glEnd();
    }
    else
    {
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 32; i++)   {
            float theta = 2.0f * 3.1415926f * float(i) / 32.0f;
            float x = radius * cosf(theta);
            float y = radius * sinf(theta);
            glVertex2f(x + center.x, y + center.y);
        }
        glEnd();
    }
}

void Engine::drawRectangle(const Point2d &center, float width, float height, float r, float g, float b, bool solid)
{
    glColor3f(r, g, b);
    float halfWidth = width * 0.5f;
    float halfHeight = height * 0.5f;
    if(solid)
    {
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(center.x-halfWidth, center.y-halfHeight);
        glVertex2f(center.x-halfWidth, center.y+halfHeight);
        glVertex2f(center.x+halfWidth, center.y+halfHeight);
        glVertex2f(center.x+halfWidth, center.y-halfHeight);
        glEnd();
    }
    else
    {
        glBegin(GL_LINE_LOOP);
        glVertex2f(center.x-halfWidth, center.y-halfHeight);
        glVertex2f(center.x-halfWidth, center.y+halfHeight);
        glVertex2f(center.x+halfWidth, center.y+halfHeight);
        glVertex2f(center.x+halfWidth, center.y-halfHeight);
        glEnd();
    }
}

void Engine::drawLine(const Point2d &p1, const Point2d &p2, float r, float g, float b)
{
    glColor3f(r, g, b);
    glBegin(GL_LINES);
        glVertex2f(p1.x, p1.y);
        glVertex2f(p2.x, p2.y);
    glEnd();
}

void Engine::drawPolyLine(const std::vector<Point2d> &points, float r, float g, float b)
{
    assert(points.size() > 1);
    glColor3f(r, g, b);
    glBegin(GL_LINE_STRIP);
    for(int i = 0;i<points.size();i++)
        glVertex2f(points[i].x, points[i].y);
    glEnd();
}

void Engine::drawPoint(const Point2d &a, float r, float g, float b)
{
    glColor3f(r, g, b);
    glBegin(GL_POINT);
        glVertex2f(a.x, a.y);
    glEnd();
}
void Engine::drawPointCloud(const std::vector<Point2d> &points, float r, float g, float b)
{
    glColor3f(r, g, b);
    glBegin(GL_POINT);
    for(int i = 0;i<points.size();i++)
        glVertex2f(points[i].x, points[i].y);
    glEnd();
}
