#ifndef __POINT_2D_H__
#define __POINT_2D_H__

#include <vector>
#include "math.h"

class Point2d
{
public:
    Point2d() { x = 0.0; y = 0.0; };
    Point2d(const Point2d &other) { this->x = other.x; this->y = other.y; };
    Point2d(double x, double y) { this->x = x; this->y = y; };
    
    Point2d& operator=(const Point2d &point) 
    {
        this->x = point.x; 
        this->y = point.y; 
        
        return *this; 
    }   
    Point2d& operator+=(const Point2d &point) 
    {
        this->x += point.x; 
        this->y += point.y; 
        
        return *this; 
    }    
    Point2d& operator-=(const Point2d &point) 
    {
        this->x -= point.x; 
        this->y -= point.y; 
        
        return *this; 
    }   
    Point2d& operator*=(const Point2d &point) 
    {
        this->x *= point.x; 
        this->y *= point.y; 
        
        return *this; 
    }
    Point2d& operator/=(const Point2d &point) 
    {
        this->x /= point.x; 
        this->y /= point.y; 
        
        return *this; 
    }
    Point2d& operator+=(double a) 
    {
        this->x += a; 
        this->y += a; 
        
        return *this; 
    }
    Point2d& operator-=(double a) 
    {
        this->x -= a; 
        this->y -= a; 
        
        return *this; 
    } 
    Point2d& operator*=(double a) 
    {
        this->x *= a; 
        this->y *= a; 
        
        return *this; 
    } 
    Point2d& operator/=(double a) 
    {
        this->x /= a; 
        this->y /= a; 
        
        return *this; 
    } 
    Point2d operator+(const Point2d &point)
    {
        return Point2d(this->x + point.x, this->y + point.y);
    }
    Point2d operator-(const Point2d &point)
    {
        return Point2d(this->x - point.x, this->y - point.y);
    }
    Point2d operator*(const Point2d &point)
    {
        return Point2d(this->x * point.x, this->y * point.y);
    }
    Point2d operator/(const Point2d &point)
    {
        return Point2d(this->x / point.x, this->y / point.y);
    }
    Point2d operator+(double a)
    {
        return Point2d(this->x + a, this->y + a);
    }
    Point2d operator-(double a)
    {
        return Point2d(this->x - a, this->y - a);
    }
    Point2d operator*(double a)
    {
        return Point2d(this->x * a, this->y * a);
    }
    Point2d operator/(double a)
    {
        return Point2d(this->x / a, this->y / a);
    }
  
    double distance(const Point2d &point) const;
    double distance2(const Point2d &point) const;
    double length() const { return distance(Point2d(0.0, 0.0)); };
    double length2() const { return distance2(Point2d(0.0, 0.0)); };
    static Point2d direction(const Point2d &origin, const Point2d &destination);
    Point2d& normalize();
    Point2d normalized() const;
    double dot(const Point2d &v) const;
    Point2d& rotateRadians(double radians);
    Point2d& rotateDegrees(double degrees);
    Point2d rotatedRadians(double radians) const;
    Point2d rotatedDegrees(double degrees) const;
    double angle(const Point2d &v) const;
    
    double x;
    double y;
};

#endif
