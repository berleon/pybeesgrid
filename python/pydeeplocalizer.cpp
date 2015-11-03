#define PY_ARRAY_UNIQUE_SYMBOL generate_data_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <deepdecoder.h>
#include <GeneratedGrid.h>
#include <boost/python.hpp>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <GroundTruthDataLoader.h>
#include <iomanip>

namespace py = boost::python;
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

template<typename N>
std::vector<N> pylistToVec(py::list pylist) {
    std::vector<N> vec;
    for (int i = 0; i < py::len(pylist); ++i)
    {
        vec.push_back(py::extract<N>(pylist[i]));
    }
    return vec;
}

void setGridParams(const GeneratedGrid & grid, double * &ptr) {
    *ptr = grid.getZRotation(); ++ptr;
    *ptr = grid.getYRotation(); ++ptr;
    *ptr = grid.getXRotation(); ++ptr;
    *ptr = grid.getCenter().x; ++ptr;
    *ptr = grid.getCenter().y; ++ptr;
    *ptr = grid.getRadius(); ++ptr;
}

PyObject * generateBatchScaled(GridArtist & artist, size_t batch_size,
                    const std::vector<GeneratedGrid> & grids,
                    const shape4d_t shape,
                    double scale) {
    shape4d_t scaled_shape = shape;
    for(size_t i = 2; i < shape.size(); i++) {
        int dim = shape.at(i);
        scaled_shape[i] = npy_intp(round(dim*scale));
    }
    uchar *raw_data = static_cast<uchar*>(calloc(get_count(scaled_shape), sizeof(uchar)));
    size_t pixels_per_tag = scaled_shape[2]*scaled_shape[3];
    for(size_t i = 0; i < batch_size; i++) {
        const GeneratedGrid & grid = grids.at(i);
        GeneratedGrid gg = grid.scale(scale);
        assert(i*pixels_per_tag < get_count(scaled_shape));
        cv::Mat mat(scaled_shape[2], scaled_shape[3], CV_8UC1, raw_data + i*pixels_per_tag);
        artist.draw(gg, mat);
        assert(mat.refcount == nullptr);
    }
    return newPyArrayOwnedByNumpy(scaled_shape, NPY_UBYTE, raw_data);
}

py::tuple generateBatch(GridGenerator & gen, GridArtist & artist, size_t batch_size, py::list py_scales) {
    auto scales = pylistToVec<double>(py_scales);
    const shape4d_t shape{static_cast<npy_intp>(batch_size), 1, TAG_SIZE, TAG_SIZE};
    std::array<npy_intp, 2> labels_shape{static_cast<npy_intp>(batch_size), Grid::NUM_MIDDLE_CELLS};
    static const size_t n_params = 6;
    std::array<npy_intp, 2> grid_params_shape{static_cast<npy_intp>(batch_size), n_params};
    float *raw_labels = static_cast<float*>(calloc(get_count(labels_shape), sizeof(float)));
    double *raw_grid_params = static_cast<double*>(
            calloc(get_count(grid_params_shape), sizeof(double)));
    float *label_ptr = raw_labels;
    double *grid_params_ptr = raw_grid_params;

    std::vector<GeneratedGrid> grids;
    for(size_t i = 0; i < batch_size; i++) {
        auto gg = gen.randomGrid();
        auto label = gg.getLabelAsVector<float>();
        memcpy(label_ptr, &label[0], label.size()*sizeof(float));
        label_ptr += label.size();
        grids.emplace_back(std::move(gg));
        setGridParams(gg, grid_params_ptr);
    }
    py::list return_py_objs;
    for(auto & scale : scales) {
        return_py_objs.append(py::handle<>(
            generateBatchScaled(artist, batch_size, grids, shape, scale)));
    }
    return_py_objs.append(py::handle<>(newPyArrayOwnedByNumpy(labels_shape, NPY_FLOAT, raw_labels)));
    return_py_objs.append(py::handle<>(newPyArrayOwnedByNumpy(grid_params_shape, NPY_DOUBLE, raw_grid_params)));
    return py::tuple(return_py_objs);
}

class PyGTDataLoader {
public:
    PyGTDataLoader(py::list pyfiles) :
            _gt_loader(pylistToVec<std::string>(pyfiles)) { };

    py::object batch(size_t batch_size, bool repeat) {
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
            return py::make_tuple(
                    py::handle<>(imagesToPyArray(batch.first, batch_size)),
                    py::handle<>(labelsToPyArray(batch.second, batch_size))
            );
        } else {
            return py::object(py::handle<>(Py_None));
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
    py::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();
    return NULL;
}
#define ATTR(NAME) py::scope().attr(#NAME) = NAME
#define ENUM_ATTR(NAME) py::scope().attr(#NAME) = size_t(NAME)

BOOST_PYTHON_MODULE(pydeepdecoder)
{
    init_numpy();
    py::def("generateBatch", generateBatch);

    ENUM_ATTR(INNER_BLACK_SEMICIRCLE);
    ENUM_ATTR(CELL_0_BLACK);
    ENUM_ATTR(CELL_1_BLACK);
    ENUM_ATTR(CELL_2_BLACK);
    ENUM_ATTR(CELL_3_BLACK);
    ENUM_ATTR(CELL_4_BLACK);
    ENUM_ATTR(CELL_5_BLACK);
    ENUM_ATTR(CELL_6_BLACK);
    ENUM_ATTR(CELL_7_BLACK);
    ENUM_ATTR(CELL_8_BLACK);
    ENUM_ATTR(CELL_9_BLACK);
    ENUM_ATTR(CELL_10_BLACK);
    ENUM_ATTR(CELL_11_BLACK);
    ENUM_ATTR(BACKGROUND_RING);
    ENUM_ATTR(IGNORE);
    ENUM_ATTR(CELL_0_WHITE);
    ENUM_ATTR(CELL_1_WHITE);
    ENUM_ATTR(CELL_2_WHITE);
    ENUM_ATTR(CELL_3_WHITE);
    ENUM_ATTR(CELL_4_WHITE);
    ENUM_ATTR(CELL_5_WHITE);
    ENUM_ATTR(CELL_6_WHITE);
    ENUM_ATTR(CELL_7_WHITE);
    ENUM_ATTR(CELL_8_WHITE);
    ENUM_ATTR(CELL_9_WHITE);
    ENUM_ATTR(CELL_10_WHITE);
    ENUM_ATTR(CELL_11_WHITE);
    ENUM_ATTR(OUTER_WHITE_RING);
    ENUM_ATTR(INNER_WHITE_SEMICIRCLE);

    ATTR(TAG_SIZE);

    py::class_<GridGenerator>("GridGenerator")
            .def("setYawAngle", &GridGenerator::setYawAngle)
            .def("setPitchAngle", &GridGenerator::setPitchAngle)
            .def("setRollAngle", &GridGenerator::setRollAngle)
            .def("setCenter", &GridGenerator::setCenter);

    py::class_<GridArtist, boost::noncopyable>("BadGridArtist", py::no_init);

    py::class_<BadGridArtist, py::bases<GridArtist>>("BadGridArtist")
            .def("setWhite", &BadGridArtist::setWhite)
            .def("setBlack", &BadGridArtist::setBlack)
            .def("setBackground", &BadGridArtist::setBackground);

    py::class_<BlackWhiteArtist, py::bases<GridArtist>>("BlackWhiteArtist");

    py::class_<MaskGridArtist, py::bases<GridArtist>>("MaskGridArtist");

    py::class_<PyGTDataLoader>("GTDataLoader", py::init<py::list>())
            .def("batch", &PyGTDataLoader::batch);
}

