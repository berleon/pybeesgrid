#pragma once

#include <opencv2/opencv.hpp> // CV_PI, cv::Matx
#include <cmath>              // std::{sin,cos}
#include <stdexcept>          // std::invalid_argument

/**
 * Computer vision helper functions
 */
namespace CvHelper
{
	/*
	 * @brief makeCanvas Makes composite image from the given images
	 * @param vecMat Vector of Images.
	 * @param windowHeight The height of the new composite image to be formed.
	 * @param nRows Number of rows of images. (Number of columns will be calculated
	 *              depending on the value of total number of images).
	 * @return new composite image.
	 */
	inline cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
		int N = vecMat.size();
		nRows  = nRows > N ? N : nRows;
		int edgeThickness = 10;
		int imagesPerRow = static_cast<int>(ceil(double(N) / nRows));
		int resizeHeight = static_cast<int>(floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness);
		int maxRowLength = 0;

		std::vector<int> resizeWidth;
		for (int i = 0; i < N;) {
				int thisRowLen = 0;
				for (int k = 0; k < imagesPerRow; k++) {
						double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
						int temp = int( ceil(resizeHeight * aspectRatio));
						resizeWidth.push_back(temp);
						thisRowLen += temp;
						if (++i == N) break;
				}
				if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
						maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
				}
		}
		int windowWidth = maxRowLength;
		cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

		for (int k = 0, i = 0; i < nRows; i++) {
				int y = i * resizeHeight + (i + 1) * edgeThickness;
				int x_end = edgeThickness;
				for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
						int x = x_end;
						cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
						cv::Mat target_ROI = canvasImage(roi);
						cv::resize(vecMat[k], target_ROI, target_ROI.size());
						x_end += resizeWidth[k] + edgeThickness;
				}
		}
		return canvasImage;
	}

	/**
	 * Convert degree to radian.
	 * @param deg degree value,
	 * @return rad value.
	 */
	inline float degToRad(float deg)
	{
		return deg * static_cast<float>(CV_PI) / 180.0f;
	}

	/**
	 * Convert radian to degree.
	 * @param rad radian value,
	 * @return degree value.
	 */
	inline float radToDeg(float rad)
	{
		return rad * 180.0f / static_cast<float>(CV_PI);
	}


	/**
	 * Roll.
	 * Performs a counterclockwise rotation of gamma about the x-axis.
	 *
	 * @see: http://planning.cs.uiuc.edu/node102.html
	 *
	 */
	inline cv::Matx<double, 3, 3> rotationMatrix_x(double gamma_rad) {
		using std::cos;
		using std::sin;

		// rotation angles:
		const double g = gamma_rad;  // angle to rotate around x axis

		return cv::Matx<double, 3, 3>
		(
		   1.0, /* | */ 0.0,    /* | */ 0.0,
		/*---------+---------------+------------*/
		   0.0, /* | */ cos(g), /* | */ -sin(g),
		/*---------+---------------+------------*/
		   0.0, /* | */ sin(g), /* | */ cos(g)
		);
	}


	/**
	 * Pitch.
	 * Performs a counterclockwise rotation of beta about the y-axis.
	 *
	 * @see: http://planning.cs.uiuc.edu/node102.html
	 *
	 */
	inline cv::Matx<double, 3, 3> rotationMatrix_y(double beta_rad) {
		using std::cos;
		using std::sin;

		// rotation angles:
		const double b = beta_rad;  // angle to rotate around y axis

		return cv::Matx<double, 3, 3>
		(
		   cos(b),  /* | */ 0.0, /* | */ sin(b),
		/*-------------+------------+-----------*/
		   0.0,     /* | */ 1.0, /* | */ 0.0,
		/*-------------+------------+-----------*/
		   -sin(b), /* | */ 0.0, /* | */ cos(b)
		);
	}


	/**
	 * Yaw.
	 * Performs a counterclockwise rotation of alpha about the z-axis.
	 *
	 * @see: http://planning.cs.uiuc.edu/node102.html
	 *
	 */
	inline cv::Matx<double, 3, 3> rotationMatrix_z(double alpha_rad) {
		using std::cos;
		using std::sin;

		// rotation angles:
		const double a = alpha_rad;  // angle to rotate around z axis

		return cv::Matx<double, 3, 3>
		(
		   cos(a), /* | */ -sin(a), /* | */ 0.0,
		/*------------+----------------+---------*/
		   sin(a), /* | */ cos(a),  /* | */ 0.0,
		/*------------+----------------+---- ----*/
		   0.0,    /* | */ 0.0,     /* | */ 1.0
		);
	}


	/**
	 * Roll-pitch-yaw.
	 * Rotates a point by roll_rad about the x-axis, then by pitch_rad about the y-axis and then by yaw_rad about the z-axis.
	 *
	 * rotationMatrix(yaw, pitch, roll) = rotationMatrix_z(yaw) * rotationMatrix_y(pitch) * rotationMatrix_x(roll)
	 *
	 * @see: http://planning.cs.uiuc.edu/node102.html
	 *
	 */
	inline cv::Matx<double, 3, 3> rotationMatrix(double yaw_rad, double pitch_rad, double roll_rad)
	{
		using std::cos;
		using std::sin;

		const double a = yaw_rad;   // angle to rotate around z axis
		const double b = pitch_rad; // angle to rotate around y axis
		const double g = roll_rad;  // angle to rotate around x axis

		return cv::Matx<double, 3, 3>
		(
		   cos(a)*cos(b), /* | */ cos(a)*sin(b)*sin(g) - sin(a)*cos(g), /* | */ cos(a)*sin(b)*cos(g) + sin(a)*sin(g),
		/*-------------------+---------------------------------------------+------------------------------------------*/
		   sin(a)*cos(b), /* | */ sin(a)*sin(b)*sin(g) + cos(a)*cos(g), /* | */ sin(a)*sin(b)*cos(g) - cos(a)*sin(g),
		/*-------------------+---------------------------------------------+------------------------------------------*/
		   -sin(b),       /* | */ cos(b)*sin(g),                        /* | */ cos(b)*cos(g)
		);
	}

	inline void drawPolyline(cv::Mat &img, std::vector<cv::Point> const &contour, const cv::Scalar& color, bool close = false, cv::Point offset = cv::Point(), int thickness = 1)
	{
	   if (contour.size() < 2) {
			   throw std::invalid_argument("a contour contains a least 2 points");
	   }
	   for (size_t i = 1; i < contour.size(); i++) {
			   cv::line(img, offset + contour[i - 1], offset + contour[i], color, thickness);
	   }
	   if (close) {
			   cv::line(img, offset + contour.back(), offset + contour.front(), color, thickness);
	   }
	}

	/**
	 *
	 * @param img            Image
	 * @param contours       All the input contours. Each contour is stored as a point vector.
	 * @param index_contour  Parameter indicating a contour to draw
	 * @param color          Color of the contours
	 * @param close          Parameter indicating if the last and first vertex of the contour should be connected
	 * @param offset         Optional contour shift parameter.
	 */
	inline void drawPolyline(cv::Mat &img, std::vector<std::vector<cv::Point>> const &contours, size_t index_contour, cv::Scalar const & color, bool close = false, cv::Point offset = cv::Point(), int thickness = 1)
	{
		drawPolyline(img, contours[index_contour], color, close, offset, thickness);
	}


}
