#pragma once

#include <bitset>
#include <cstdint>
#include <vector>
#include <pipeline/common/Grid.h>

#include "../datastructure/Ellipse.h"

namespace Util {

typedef struct {
	cv::Point2i center;
	double radius;
	double angle_z;
	double angle_y;
	double angle_x;
} gridconfig_t;


std::array<gridconfig_t, 2> gridCandidatesFromEllipse(const pipeline::Ellipse& ellipse, const double rotation = 0,
													  const double focal_length = Grid::Structure::DEFAULT_FOCAL_LENGTH);

inline bool pointInBounds(cv::Rect const& bounds, cv::Point const& point) {
	return (point.x >= bounds.tl().x &&
	        point.y >= bounds.tl().y &&
	        point.x <  bounds.br().x &&
	        point.y <  bounds.br().y);
}

inline bool pointInBounds(cv::Size const& size, cv::Point const& point) {
	return pointInBounds(cv::Rect(0, 0, size.width, size.height), point);
}

// branchless, type-safe signum
// see: http://stackoverflow.com/a/4609795
template <typename T>
int sgn(T val) {
		return (T(0) < val) - (val < T(0));
}

template <typename T>
std::vector<T> linspace(T first, T last, size_t len) {
		std::vector<T> result(len);
		T step = (last-first) / (len - 1);
		for (size_t i=0; i<len; i++) { result[i] = first + static_cast<T>(i) * step; }
		return result;
}

template <std::size_t N>
inline void
rotateBitset(std::bitset<N>& b, unsigned m)
{
		b = b << m | b >> (N-m);
}
}
