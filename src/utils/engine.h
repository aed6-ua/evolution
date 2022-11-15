#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <vector>
#include "point2d.h"

class Engine
{
public:
    Engine();
    ~Engine();
    
    void drawCircle(const Point2d &center, float radius, float r, float g, float b, bool solid = false);
    void drawRectangle(const Point2d &center, float width, float height, float r, float g, float b, bool solid = false);
    void drawLine(const Point2d &p1, const Point2d &p2, float r, float g, float b);
    void drawPolyLine(const std::vector<Point2d> &points, float r, float g, float b);
    void drawPoint(const Point2d &a, float r, float g, float b);
    void drawPointCloud(const std::vector<Point2d> &points, float r, float g, float b);
};

#endif
