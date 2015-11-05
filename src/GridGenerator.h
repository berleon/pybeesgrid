
#pragma once

#include <bits/unique_ptr.h>
#include "deepdecoder.h"
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

namespace deepdecoder {

class GridGenerator {

public:
    GridGenerator();
    GridGenerator(unsigned long seed);
    GeneratedGrid randomGrid();
    UNIFORM_REAL_DISTRIBUTION_MEMBER(YawAngle)
    UNIFORM_REAL_DISTRIBUTION_MEMBER(PitchAngle)
    UNIFORM_REAL_DISTRIBUTION_MEMBER(RollAngle)
    UNIFORM_INT_DISTRIBUTION_MEMBER(Radius)
    NORMAL_DISTRIBUTION_MEMBER(Center)
public:
    std::unique_ptr<GridGenerator> clone();
private:
    Grid::idarray_t generateID();
    std::mt19937_64 _re;
    std::bernoulli_distribution _coin_dis;
};
}
