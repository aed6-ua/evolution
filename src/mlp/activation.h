#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <math.h>

namespace activation {

    inline double sigmoid(double x)
    {
        return 1 / (1 + exp(-x));
    }
    
    inline double tanH(double x)
    {
        return (tanh)(x);
    }
    
    inline double linear(double x)
    {
        return x;
    }
    
    inline double relu(double x)
    {
        return x < 0.0 ? 0.0 : x;
    }
}

#endif
