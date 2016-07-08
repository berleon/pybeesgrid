#pragma once

#include <boost/logic/tribool.hpp>
#include <opencv2/core/core.hpp>

#include <cstddef>
#include <array>
#include <memory>


class Grid {
public:
	// number of cells around the center semicircles
	static const size_t NUM_MIDDLE_CELLS = 12;

	// indices in polygon vector
	static const size_t INDEX_OUTER_WHITE_RING       = 0;
	static const size_t INDEX_INNER_WHITE_SEMICIRCLE = 1;
	static const size_t INDEX_INNER_BLACK_SEMICIRCLE = 2;
	static const size_t INDEX_INNER_LINE             = 3;
	static const size_t INDEX_MIDDLE_CELLS_BEGIN     = 4;
	static const size_t INDEX_MIDDLE_CELLS_END       = INDEX_MIDDLE_CELLS_BEGIN + NUM_MIDDLE_CELLS;

	// total number of cells (non-coding and coding)
	static const size_t NUM_CELLS = INDEX_MIDDLE_CELLS_END;

	static const size_t POINTS_PER_MIDDLE_CELL = 3;
	static const size_t POINTS_PER_LINE = 3;
	static_assert(POINTS_PER_LINE % 2 != 0, "POINTS_PER_LINE must be odd");
	static const size_t POINTS_PER_RING = NUM_MIDDLE_CELLS * POINTS_PER_MIDDLE_CELL;

	static_assert(POINTS_PER_RING % 4 == 0 , "POINTS_PER_RING = NUM_MIDDLE_CELLS * POINTS_PER_MIDDLE_CELL must be a multiple of 4");

	typedef std::array<boost::tribool, NUM_MIDDLE_CELLS> idarray_t;

	explicit Grid(cv::Point2i center, double radius, double angle_z, double angle_y, double angle_x);

	virtual ~Grid() {}

	void setXRotation(double angle);
	double getXRotation() const { return _angle_x; }

	void setYRotation(double angle);
	double getYRotation() const { return _angle_y; }

	void setZRotation(double angle);
	double getZRotation() const { return _angle_z; }

	void setCenter(cv::Point c);
	cv::Point getCenter() const { return _center; }

	void setRadius(double radius);
	double getRadius() const { return _radius; }

	idarray_t const& getIdArray() const { return _ID; }

	/**
	 * Axis-aligned minimum bounding box of the grid
	 */
	cv::Rect getBoundingBox() const;

	/**
	 * Axis-aligned minimum bounding box of the grid centered at (0, 0)
	 */
	cv::Rect getOriginBoundingBox() const {	return _boundingBox; }

    struct Structure {
        static const double DEFAULT_INNER_RING_RADIUS;
        static const double DEFAULT_MIDDLE_RING_RADIUS;
        static const double DEFAULT_OUTER_RING_RADIUS;
        static const double DEFAULT_BULGE_FACTOR;
        static const double DEFAULT_FOCAL_LENGTH;

        const double INNER_RING_RADIUS;
        const double MIDDLE_RING_RADIUS;
        const double OUTER_RING_RADIUS;
        const double BULGE_FACTOR;
        const double FOCAL_LENGTH;

        Structure(double inner_ring_radius, double middle_ring_radius, double outer_ring_radius,
                  double bulge_factor, double focal_length)
                : INNER_RING_RADIUS(inner_ring_radius), MIDDLE_RING_RADIUS(middle_ring_radius),
                  OUTER_RING_RADIUS(outer_ring_radius), BULGE_FACTOR(bulge_factor), FOCAL_LENGTH(focal_length)
        { };
    };

	static inline double z_coordinate(const double radius, const double bugle_factor) {
		return - std::cos(bugle_factor * radius);
	}

	inline cv::Point2i projectPoint(cv::Point3d p) const {
		const double f = _structure->FOCAL_LENGTH;
		return cv::Point2i(static_cast<int>(round((p.x / (p.z + f))  * _radius * f)),
						   static_cast<int>(round((p.y / (p.z + f)) * _radius * f)));
	}

    explicit Grid(cv::Point2i center, double radius, double angle_z, double angle_y, double angle_x,
                  const std::shared_ptr<Grid::Structure> structure);
protected:

	enum RingIndex {
		INNER_RING = 0,
		MIDDLE_RING,
		OUTER_RING
	};

	template<typename POINT>
	struct coordinates_t
	{
		typedef typename POINT::value_type              value_type;
		typedef POINT                                   point_type;
		typedef std::array<point_type, POINTS_PER_RING> container_type;

		std::array<container_type, 3> _rings;
		std::array<point_type, POINTS_PER_LINE> _inner_line;

		container_type &_inner_ring;
		container_type &_middle_ring;
		container_type &_outer_ring;

		// default constructor with member initialization
		coordinates_t()
			: _inner_ring(_rings[INNER_RING])
			, _middle_ring(_rings[MIDDLE_RING])
			, _outer_ring(_rings[OUTER_RING])
		{}

		// move constructor, && : r-value reference
		coordinates_t(coordinates_t &&rhs)
			: _rings(std::move(rhs._rings))
			, _inner_line(std::move(rhs._inner_line))
			, _inner_ring(_rings[INNER_RING])
			, _middle_ring(_rings[MIDDLE_RING])
			, _outer_ring(_rings[OUTER_RING])
		{}

		// delete copy constructor and assignment operator
		// -> make struct non-copyable
		coordinates_t(const coordinates_t&) = delete;
		coordinates_t& operator=(const coordinates_t&) = delete;
	};

	typedef coordinates_t<cv::Point3d> coordinates3D_t;
	typedef coordinates_t<cv::Point2i> coordinates2D_t;

    /* use only for deserialization purposes! */
    explicit Grid() : _structure(_default_structure)
            , _coordinates3D(_default_coordinates3D)
            , _coordinates2D(NUM_CELLS) {}
    explicit Grid(cv::Point2i center, double radius, double angle_z, double angle_y, double angle_x,
                  const std::shared_ptr<Grid::Structure> structure,
                  const std::shared_ptr<Grid::coordinates3D_t> coordinates3D);

    /**
     * @brief precompute set of 3D points which will be transformed according to grid parameters
     */
    static Grid::coordinates3D_t generate_3D_base_coordinates(const Grid::Structure & s);

	/**
	 * @brief rotates and scales the base mesh according to given parameter set
	 */
	virtual coordinates2D_t generate_3D_coordinates_from_parameters_and_project_to_2D();

	/**
	 * @brief performs all computations required to draw the tag
	 *
	 * has to be called whenever a parameter of the grid is changed (except _center)
	 * i.e. in the constructor and in most of the setters (except setCenter ...)
	 */
	void prepare_visualization_data();

	/**
	 * Each of the following parameters define the tag uniquely. The InteractiveGrid class holds a
	 * set of 3D-points that make up a mesh of the bee tag. These points are transformed according
	 * to the given parameters and projected onto the camera plane to produce 2D coordinates for
	 * displaying the tag.
	 */
    static const std::shared_ptr<coordinates3D_t> _default_coordinates3D; // underlying 3D coordinates of grid mesh
    static const std::shared_ptr<Grid::Structure> _default_structure;

    const std::shared_ptr<Grid::Structure> _structure;   // structure of the grid, (FOCAL_LENGTH, *_RADUIS, ...)
    const std::shared_ptr<coordinates3D_t> _coordinates3D; // 3D coordinates of the mesh
	std::vector<std::vector<cv::Point>> _coordinates2D; // 2D coordinates of mesh (after perspective projection) (see opencv function drawContours)
	cv::Point2i                         _center;        // center point of the grid (within image borders - unit: px)
	double                              _radius;        // radius of the tag (unit: px)
	double                              _angle_z;       // the angle of the grid (unit: rad. points towards the head of the bee, positive is counter-clock)
	double                              _angle_y;       // the rotation angle of the grid around y axis (rotates into z - space)
	double                              _angle_x;       // the rotation angle of the grid around x axis (rotates into z - space)
	idarray_t                           _ID;            // bit pattern of tag (false and true for black and white, indeterminate for unrecognizable)
	cv::Rect                            _boundingBox;   // bounding box of the projected 2d points
};
