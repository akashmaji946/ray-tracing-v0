#ifndef INTERVAL_H
#define INTERVAL_H

#include "commons.hpp"
#include "utility.hpp"

class Interval {
public:
    double min;
    double max; 

    Interval() : min(0.0), max(0.0) {}
    Interval(double min = 0.0, double max = INF) : min(min), max(max) {}
    Interval(const Interval& other) : min(other.min), max(other.max) {}

    double size() const {
        return max - min;
    }

    bool contains(double value) const {
        return value >= min && value <= max;
    }

    bool surrounds(const Interval& other) const {
        return min <= other.min && max >= other.max;
    }

    double clamp(double value) const {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

};

const Interval EMPTY_INTERVAL = Interval(INF, -INF);
const Interval FULL_INTERVAL = Interval(-INF, INF);
inline ostream& operator<<(ostream& out, const Interval& interval) {
    return out << "[" << interval.min << ", " << interval.max << "]";
}



#endif // INTERVAL_H