#define PY_ARRAY_UNIQUE_SYMBOL generate_data_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <deepdecoder.h>
#include <GeneratedGrid.h>
#include <boost/python.hpp>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <GroundTruthDataLoader.h>
#include <iomanip>

namespace bp = boost::python;
using namespace deepdecoder;

template<size_t N>
using shape_t = std::array<npy_intp, N>;

using shape4d_t = shape_t<4>;
using shape2d_t = shape_t<2>;

template<size_t N>
size_t get_count(const shape_t<N> & dims) {
    size_t count = 1;
    for(size_t i = 0; i < dims.size(); i++) {
        count *= dims[i];
    };
    return count;
}

template<size_t N>
PyObject * newPyArrayOwnedByNumpy(shape_t<N> shape, int type, void * data) {
    PyObject * pyObj = PyArray_SimpleNewFromData(shape.size(), shape.data(), type, data);
    PyArrayObject * pyArr = reinterpret_cast<PyArrayObject *>(pyObj);
    PyArray_ENABLEFLAGS(pyArr, NPY_ARRAY_OWNDATA);
    return pyObj;
}
void setGridParams(const GeneratedGrid & grid, double * &ptr) {
    *ptr = grid.getZRotation(); ++ptr;
    *ptr = grid.getYRotation(); ++ptr;
    *ptr = grid.getXRotation(); ++ptr;
    *ptr = grid.getCenter().x; ++ptr;
    *ptr = grid.getCenter().y; ++ptr;
    *ptr = grid.getRadius(); ++ptr;
}
bp::tuple generateBatch(GridGenerator & gen, GridArtist & artist, size_t batch_size) {
    const shape4d_t shape{static_cast<npy_intp>(batch_size), 1, TAG_SIZE, TAG_SIZE};
    std::array<npy_intp, 2> labels_shape{static_cast<npy_intp>(batch_size), Grid::NUM_MIDDLE_CELLS};
    static const size_t n_params = 6;
    std::array<npy_intp, 2> grid_params_shape{static_cast<npy_intp>(batch_size), n_params};
    uchar *raw_data = static_cast<uchar*>(calloc(get_count(shape), sizeof(uchar)));
    float *raw_labels = static_cast<float*>(calloc(get_count(labels_shape), sizeof(float)));
    double *raw_grid_params = static_cast<double*>(
            calloc(get_count(grid_params_shape), sizeof(double)));
    float *label_ptr = raw_labels;
    double *grid_params_ptr = raw_grid_params;
    for(size_t i = 0; i < batch_size; i++) {
        GeneratedGrid gg = gen.randomGrid();
        cv::Mat mat(TAG_SIZE, TAG_SIZE, CV_8UC1, raw_data + i*TAG_PIXELS);
        artist.draw(gg, mat);
        auto label = gg.getLabelAsVector<float>();
        memcpy(label_ptr, &label[0], label.size()*sizeof(float));
        label_ptr += label.size();
        CHECK(mat.refcount == nullptr);
        setGridParams(gg, grid_params_ptr);
    }
    return bp::make_tuple(
            bp::handle<>(newPyArrayOwnedByNumpy(shape, NPY_UBYTE, raw_data)),
            bp::handle<>(newPyArrayOwnedByNumpy(labels_shape, NPY_FLOAT, raw_labels)),
            bp::handle<>(newPyArrayOwnedByNumpy(grid_params_shape, NPY_DOUBLE, raw_grid_params))
    );
}

template<typename N>
std::vector<N> pylistToVec(bp::list pylist) {
    std::vector<N> vec;
    for (int i = 0; i < bp::len(pylist); ++i)
    {
        vec.push_back(bp::extract<N>(pylist[i]));
    }
    return vec;
}

class PyGTDataLoader {
public:
    PyGTDataLoader(bp::list pyfiles) :
            _gt_loader(pylistToVec<std::string>(pyfiles)) { };

    bp::object batch(size_t batch_size, bool repeat) {
        GTRepeat repeat_enum;
        if (repeat) {
            repeat_enum = REPEAT;
        } else {
            repeat_enum = NO_REPEAT;
        }
        boost::optional<dataset_t<float>> opt_batch = _gt_loader.batch(batch_size,
                                                                       repeat_enum);
        if(opt_batch) {
            auto batch = opt_batch.get();
            return bp::make_tuple(
                    bp::handle<>(imagesToPyArray(batch.first, batch_size)),
                    bp::handle<>(labelsToPyArray(batch.second, batch_size))
            );
        } else {
            return bp::object(bp::handle<>(Py_None));
        }
    };

private:
    PyObject * imagesToPyArray(std::vector<cv::Mat> images, size_t batch_size) {
        std::array<npy_intp, 4> images_shape{static_cast<npy_intp>(batch_size), 1, TAG_SIZE, TAG_SIZE};
        size_t image_count = get_count<4>(images_shape);
        uchar *raw_img_data = static_cast<uchar*>(calloc(image_count, sizeof(uchar)));
        uchar *iter_ptr = raw_img_data;
        for(auto & mat : images) {
            for(long j = 0; j < mat.rows; j++) {
                memcpy(iter_ptr, mat.ptr(j), mat.cols);
                iter_ptr += mat.cols;
            }
        }
        return newPyArrayOwnedByNumpy(images_shape, NPY_UBYTE, raw_img_data);
    }

    PyObject * labelsToPyArray(std::vector<std::vector<float>> labels, size_t batch_size) {
        std::stringstream ss;
        for(size_t i = 0; i < labels.size(); i++) {
            auto & label = labels.at(i);
            ss << "#" << std::setw(3) << i  << ": ";
            for(auto & bit : label) {
                if(bit == 0) {
                    ss << '#';
                } else {
                    ss << ' ';
                }
            }
            ss << '\n';
        }
        // std::cout << ss.str() << std::endl;
        std::array<npy_intp, 2> labels_shape{static_cast<npy_intp>(batch_size), Grid::NUM_MIDDLE_CELLS};
        size_t labels_count = get_count<2>(labels_shape);
        float *raw_label_data = static_cast<float*>(calloc(labels_count, sizeof(float)));
        float *label_ptr = raw_label_data;
        for(auto & label : labels) {
            memcpy(label_ptr, &label[0], label.size()*sizeof(float));
            label_ptr += label.size();
        }
        return newPyArrayOwnedByNumpy(labels_shape, NPY_FLOAT32, raw_label_data);
    }

    GroundTruthDataLoader<float> _gt_loader;
};

void * init_numpy() {
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();
    return NULL;
}
#define ATTR(NAME) \
    bp::scope().attr(#NAME) = #NAME

BOOST_PYTHON_MODULE(pydeepdecoder)
{
    init_numpy();
    bp::def("generateBatch", generateBatch);
    ATTR(INNER_BLACK_SEMICIRCLE);
    ATTR(CELL_0_BLACK);
    ATTR(CELL_1_BLACK);
    ATTR(CELL_2_BLACK);
    ATTR(CELL_3_BLACK);
    ATTR(CELL_4_BLACK);
    ATTR(CELL_5_BLACK);
    ATTR(CELL_6_BLACK);
    ATTR(CELL_7_BLACK);
    ATTR(CELL_8_BLACK);
    ATTR(CELL_9_BLACK);
    ATTR(CELL_10_BLACK);
    ATTR(CELL_11_BLACK);
    ATTR(IGNORE);
    ATTR(CELL_0_WHITE);
    ATTR(CELL_1_WHITE);
    ATTR(CELL_2_WHITE);
    ATTR(CELL_3_WHITE);
    ATTR(CELL_4_WHITE);
    ATTR(CELL_5_WHITE);
    ATTR(CELL_6_WHITE);
    ATTR(CELL_7_WHITE);
    ATTR(CELL_8_WHITE);
    ATTR(CELL_9_WHITE);
    ATTR(CELL_10_WHITE);
    ATTR(CELL_11_WHITE);
    ATTR(OUTER_WHITE_RING);
    ATTR(INNER_WHITE_SEMICIRCLE);

    bp::class_<GridGenerator>("GridGenerator")
            .def("setYawAngle", &GridGenerator::setYawAngle)
            .def("setPitchAngle", &GridGenerator::setPitchAngle)
            .def("setRollAngle", &GridGenerator::setRollAngle)
            .def("setCenter", &GridGenerator::setCenter);

    bp::class_<GridArtist, boost::noncopyable>("BadGridArtist", bp::no_init);

    bp::class_<BadGridArtist, bp::bases<GridArtist>>("BadGridArtist")
            .def("setWhite", &BadGridArtist::setWhite)
            .def("setBlack", &BadGridArtist::setBlack)
            .def("setBackground", &BadGridArtist::setBackground);

    bp::class_<MaskGridArtist, bp::bases<GridArtist>>("MaskGridArtist");

    bp::class_<PyGTDataLoader>("GTDataLoader", bp::init<bp::list>())
            .def("batch", &PyGTDataLoader::batch);
}

