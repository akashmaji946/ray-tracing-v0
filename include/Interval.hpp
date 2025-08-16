#ifndef INTERVAL_H
#define INTERVAL_H

#include "commons.hpp"
#include "utility.hpp"

class Interval {
public:
    double min;
    double max; 

    __host__ __device__
    Interval() : min(0.0), max(0.0) {}
    __host__ __device__
    Interval(double min = 0.0, double max = INF) : min(min), max(max) {}
    __host__ __device__
    Interval(const Interval& other) : min(other.min), max(other.max) {}

    __host__ __device__
    double size() const {
        return max - min;
    }

    __host__ __device__
    bool contains(double value) const {
        return value >= min && value <= max;
    }

    __host__ __device__
    bool surrounds(const Interval& other) const {
        return min <= other.min && max >= other.max;
    }

    __host__ __device__
    double clamp(double value) const {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

};


const Interval EMPTY_INTERVAL = Interval(INF, -INF);
const Interval FULL_INTERVAL = Interval(-INF, INF);

__host__ __device__
inline ostream& operator<<(ostream& out, const Interval& interval) {
    return out << "[" << interval.min << ", " << interval.max << "]";
}

#endif // INTERVAL_H