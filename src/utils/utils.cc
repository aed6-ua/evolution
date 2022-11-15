#include "utils.h"

/// <summary>
/// Gets a random number between specified minimum and maximum values.
/// </summary>
/// <param name="min">The minimum value.</param>
/// <param name="max">The maximum value.</param>
/// <returns>
/// A random double between min and max values.
/// </returns> 
double randomNumber(const double& min, const double& max)
{
    static thread_local std::mt19937* generator = nullptr;

    std::hash<std::thread::id> hasher;
    if (!generator) generator = new std::mt19937(hasher(std::this_thread::get_id()));
    std::uniform_real_distribution<double> distribution(min, max);
    
    return distribution(*generator);
}
