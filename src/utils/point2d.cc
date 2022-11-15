#include "point2d.h"
#include <math.h>

constexpr double eps = 1e-14;
constexpr double deg2Rad = M_PI/180.0;

double Point2d::distance(const Point2d &point) const
{
    return std::sqrt(distance2(point));
}
double Point2d::distance2(const Point2d &point) const
{
    double diffx = this->x - point.x;
    double diffy = this->y - point.y;
    
    return diffx*diffx + diffy*diffy;
}
Point2d Point2d::direction(const Point2d &origin, const Point2d &destination)
{
    return Point2d(destination.x - origin.x, destination.y - origin.y).normalize();
}
Point2d& Point2d::normalize()
{
    double length = this->length();
    
    if(length != 0.0)
    {
        this->x /= length;
        this->y /= length;
    }
    else
    {
        this->x = 0.0;
        this->y = 0.0;
    }
    
    return *this;
}

Point2d Point2d::normalized() const
{
    return Point2d(*this).normalize();
}

double Point2d::dot(const Point2d &point) const
{
    return this->x * point.x + this->y * point.y;
}

Point2d& Point2d::rotateRadians(double radians)
{
    double ca = std::cos(radians);
    double sa = std::sin(radians);
    
    double newX = x * ca - y * sa;
    double newY = x * sa + y * ca;
    
    x = newX;
    y = newY;
        
    return *this;
}
Point2d& Point2d::rotateDegrees(double degrees)
{
    return rotateRadians(degrees * deg2Rad);
}

Point2d Point2d::rotatedRadians(double radians) const
{
    return Point2d(*this).rotateRadians(radians);
}
Point2d Point2d::rotatedDegrees(double degrees) const
{
    return Point2d(*this).rotateRadians(degrees * deg2Rad);
}

double Point2d::angle(const Point2d &v) const
{
    double s = std::sqrt(length2() * v.length2());
	if(s != 0.0)
	    return std::acos(dot(v) / s);
	else
	    return 0.0;
}
