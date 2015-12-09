
#pragma once

#include <bits/unique_ptr.h>
#include "beesgrid.h"
#include "GeneratedGrid.h"

#define DISTRIBUTION_MEMBER(NAME, DISTRIBUTION_T, TYPE) \
public:\
    inline void set##NAME(TYPE begin, TYPE end) { \
        _##NAME = std::make_pair(begin, end); \
        _##NAME##_dis = DISTRIBUTION_T(begin, end); \
    } \
    inline std::pair<TYPE, TYPE> get##NAME() const { \
        return _##NAME; \
    }; \
private: \
    std::pair<TYPE, TYPE> _##NAME; \
    DISTRIBUTION_T _##NAME##_dis;

#define UNIFORM_INT_DISTRIBUTION_MEMBER(NAME) DISTRIBUTION_MEMBER( NAME, std::uniform_int_distribution<>, long)
#define UNIFORM_REAL_DISTRIBUTION_MEMBER(NAME) DISTRIBUTION_MEMBER(NAME, std::uniform_real_distribution<>, double)
#define NORMAL_DISTRIBUTION_MEMBER(NAME) DISTRIBUTION_MEMBER(NAME, std::normal_distribution<>, double)

namespace beesgrid {

class GridGenerator {

public:
    GridGenerator();
    GridGenerator(unsigned long seed);
    GeneratedGrid randomGrid();
    std::unique_ptr<GridGenerator> clone();

    inline void setRollAngle(double mean, double lambda) {
        _RollAngle = std::make_pair(mean, lambda);
        _roll_angle_dis = std::exponential_distribution<>(lambda);
    }
    inline std::pair<double, double> getRollAngle() {
        return _RollAngle;
    }
    inline double sampleRollAngle() {
        const double mean = _RollAngle.first;
        const double sign = _coin_dis(_re) ? -1 : 1;
        return sign*_roll_angle_dis(_re)/2 - mean;
    }

    UNIFORM_REAL_DISTRIBUTION_MEMBER(YawAngle)
    NORMAL_DISTRIBUTION_MEMBER(PitchAngle)

    NORMAL_DISTRIBUTION_MEMBER(Radius)
    NORMAL_DISTRIBUTION_MEMBER(Center)
private:
    Grid::idarray_t generateID();
    std::mt19937_64 _re;
    std::bernoulli_distribution _coin_dis;
    std::pair<double, double> _RollAngle;
    std::exponential_distribution<> _roll_angle_dis;
};
}
