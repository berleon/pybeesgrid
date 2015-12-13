#define PY_ARRAY_UNIQUE_SYMBOL generate_data_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <beesgrid.h>
#include <GeneratedGrid.h>
#include <GridArtist.h>
#include <GridGenerator.h>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include <GroundTruthDataLoader.h>
#include <iomanip>
#include <thread>
#include <mutex>

namespace py = boost::python;
using namespace beesgrid;

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
std::vector<N> vectorFromPyList(py::list pylist) {
    std::vector<N> vec;
    for (int i = 0; i < py::len(pylist); ++i)
    {
        vec.push_back(py::extract<N>(pylist[i]));
    }
    return vec;
}

inline void setGridParams(const GeneratedGrid & grid, double * &ptr) {
    *ptr = grid.getZRotation(); ++ptr;
    *ptr = grid.getYRotation(); ++ptr;
    *ptr = grid.getXRotation(); ++ptr;
    *ptr = grid.getCenter().x; ++ptr;
    *ptr = grid.getCenter().y; ++ptr;
    *ptr = grid.getRadius(); ++ptr;
}
class ScopedGILRelease
{
public:
    inline ScopedGILRelease()
    {
        m_thread_state = PyEval_SaveThread();
    }

    inline ~ScopedGILRelease()
    {
        PyEval_RestoreThread(m_thread_state);
        m_thread_state = NULL;
    }
private:
    PyThreadState * m_thread_state;
};

void drawGridsParallelWorkFn(std::unique_ptr<GridArtist> artist,
                             const std::vector<GeneratedGrid> grids,
                             const shape4d_t scaled_shape,
                             uchar *raw_data,
                             double scale) {
    const size_t pixels_per_tag = scaled_shape[2]*scaled_shape[3];
    size_t i = 0;
    for(const auto & grid : grids) {
        GeneratedGrid gg = grid.scale(scale);
        cv::Mat mat(scaled_shape[2], scaled_shape[3], CV_8UC1, raw_data + i*pixels_per_tag);
        artist->draw(gg, mat);
        ++i;
    }
}

PyObject * drawGridsParallel(
        const GridArtist &artist,
        const std::vector<std::vector<GeneratedGrid>> &grids_vecs,
        const shape4d_t shape,
        double scale
) {
    ScopedGILRelease gil_release;

    shape4d_t scaled_shape = shape;
    for(size_t i = 2; i < shape.size(); i++) {
        int dim = shape.at(i);
        scaled_shape[i] = npy_intp(round(dim*scale));
    }
    const size_t pixels_per_tag = scaled_shape[2]*scaled_shape[3];

    uchar *raw_data = static_cast<uchar*>(calloc(get_count(scaled_shape), sizeof(uchar)));
    std::vector<std::thread> threads;
    size_t start = 0;
    for (size_t i = 0; i < grids_vecs.size(); i++) {
        uchar * raw_data_offset = raw_data + start*pixels_per_tag;
        auto & grids = grids_vecs.at(i);
        threads.push_back(std::thread(&drawGridsParallelWorkFn, artist.clone(), grids, scaled_shape, raw_data_offset, scale));
        start += grids.size();
    }
    for(auto & t : threads) {
        t.join();
    }
    return newPyArrayOwnedByNumpy(scaled_shape, NPY_UBYTE, raw_data);
}

void generateGridsParallelWorkFn(
        std::unique_ptr<GridGenerator> gen,
        const size_t nb_todo,
        std::vector<GeneratedGrid> &grids)
{
    for(size_t i = 0; i < nb_todo; i++) {
        grids.emplace_back(gen->randomGrid());
    }
}

void generateGridsParallel(
        GridGenerator &gen,
        const size_t batch_size,
        std::vector<std::vector<GeneratedGrid>> &grids_vecs
) {
    ScopedGILRelease gil_release;
    size_t nb_cpus = 2*std::thread::hardware_concurrency();
    if (nb_cpus == 0) { nb_cpus = 1; }
    std::vector<std::thread> threads;
    const size_t part = batch_size / nb_cpus;
    for (size_t i = 0; i < nb_cpus; i++) {
        grids_vecs.push_back(std::vector<GeneratedGrid>());
    }
    for (size_t i = 0; i < nb_cpus; i++) {
        const size_t start = part * i;
        size_t end;
        if (i + 1 == nb_cpus) {
            end = batch_size;
        } else {
            end = start + part;
        }
        const size_t nb_todo = end - start;
        threads.push_back(std::thread(&generateGridsParallelWorkFn, gen.clone(),
                                      nb_todo, std::ref(grids_vecs.at(i))));
    }
    for(auto & t : threads) {
        t.join();
    }
}

py::list generateBatch(GridGenerator & gen, const GridArtist & artist, const size_t batch_size, py::list py_scales) {
    auto scales = vectorFromPyList<double>(py_scales);
    const shape4d_t shape{static_cast<npy_intp>(batch_size), 1, TAG_SIZE, TAG_SIZE};
    std::array<npy_intp, 2> labels_shape{static_cast<npy_intp>(batch_size), Grid::NUM_MIDDLE_CELLS};
    static const size_t n_params = 6;
    std::array<npy_intp, 2> grid_params_shape{static_cast<npy_intp>(batch_size), n_params};
    float *raw_labels = static_cast<float*>(malloc(get_count(labels_shape)*sizeof(float)));
    double *raw_grid_params = static_cast<double*>(
            malloc(get_count(grid_params_shape) * sizeof(double)));
    float *label_ptr = raw_labels;
    double *grid_params_ptr = raw_grid_params;

    std::vector<std::vector<GeneratedGrid>> grids_vecs;
    generateGridsParallel(gen, batch_size, grids_vecs);

    for(auto & grids : grids_vecs) {
        for(auto & gg : grids) {
            auto label = gg.getLabelAsVector<float>();
            memcpy(label_ptr, &label[0], label.size()*sizeof(float));
            label_ptr += label.size();
            setGridParams(gg, grid_params_ptr);
        }
    }
    py::list return_py_objs;
    for(auto & scale : scales) {
        return_py_objs.append(py::handle<>(
                drawGridsParallel(artist, grids_vecs, shape, scale)));
    }
    return_py_objs.append(py::handle<>(newPyArrayOwnedByNumpy(labels_shape, NPY_FLOAT, raw_labels)));
    return_py_objs.append(py::handle<>(newPyArrayOwnedByNumpy(grid_params_shape, NPY_DOUBLE, raw_grid_params)));
    return return_py_objs;
}

void valueError(const std::string & msg) {
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        py::throw_error_already_set();
}

void buildGridsFromNpArrWorkFn(
        PyArrayObject * bits_and_config_ptr,
        const size_t offset,
        const size_t nb_todo,
        std::vector<GeneratedGrid> &grids)
{
    for(size_t i = offset; i < offset + nb_todo; i++) {
        Grid::idarray_t id;
        int pos = 0;
        for(size_t c = 0; c < Grid::NUM_MIDDLE_CELLS; c++) {
            float cell = *reinterpret_cast<float*>(PyArray_GETPTR2(bits_and_config_ptr, i, pos));
            ++pos;
            if(cell == 1.) {
                id[c] = true;
            } else if(cell == 0) {
                id[c] = false;
            } else {
                id[c] = boost::tribool::indeterminate_value;
            }
        }

        double rot_x =  *reinterpret_cast<float*>(PyArray_GETPTR2(bits_and_config_ptr, i, pos++));
        double rot_y =  *reinterpret_cast<float*>(PyArray_GETPTR2(bits_and_config_ptr, i, pos++));
        double rot_z =  *reinterpret_cast<float*>(PyArray_GETPTR2(bits_and_config_ptr, i, pos++));
        float x =       *reinterpret_cast<float*>(PyArray_GETPTR2(bits_and_config_ptr, i, pos++));
        float y =       *reinterpret_cast<float*>(PyArray_GETPTR2(bits_and_config_ptr, i, pos++));
        double radius = *reinterpret_cast<float*>(PyArray_GETPTR2(bits_and_config_ptr, i, pos++));
        cv::Point2i center{static_cast<int>(x), static_cast<int>(y)};
        std::cout << "[" <<  i << "]" <<
                "rot_x"  << rot_x <<
                "rot_y"  << rot_y  <<
                "rot_z"  << rot_z  <<
                "x"      << x      <<
                "y"      << y      <<
                "radius" << radius <<
                "center" << center << std::endl;

        grids.emplace_back(id, center, radius, rot_x, rot_y, rot_z);
    }
}

void buildGridsFromNpArr(PyArrayObject * bits_and_config_ptr,
                         const size_t batch_size,
                         std::vector<std::vector<GeneratedGrid>> &grids_vecs) {
    ScopedGILRelease gil_release;
    size_t nb_cpus = 2*std::thread::hardware_concurrency();
    if (nb_cpus == 0) { nb_cpus = 1; }
    std::vector<std::thread> threads;
    const size_t part = batch_size / nb_cpus;
    for (size_t i = 0; i < nb_cpus; i++) {
        grids_vecs.push_back(std::vector<GeneratedGrid>());
    }
    for (size_t i = 0; i < nb_cpus; i++) {
        const size_t start = part * i;
        size_t end;
        if (i + 1 == nb_cpus) {
            end = batch_size;
        } else {
            end = start + part;
        }
        const size_t nb_todo = end - start;
        threads.push_back(std::thread(&buildGridsFromNpArrWorkFn, bits_and_config_ptr, start,
                                      nb_todo, std::ref(grids_vecs.at(i))));
    }
    for(auto & t : threads) {
        t.join();
    }
}

py::list drawGrids(
        PyObject * bits_and_configs,
        const GridArtist & artist,
        py::list py_scales) {
    PyArrayObject* arr_ptr = reinterpret_cast<PyArrayObject*>(bits_and_configs);

    int ndims = PyArray_NDIM(arr_ptr);
    npy_intp * shape =  PyArray_SHAPE(arr_ptr);
    if (ndims != 2) {
        std::stringstream ss;
        ss << "bits_and_configs has wrong number of dimensions, expected 2 found: " << ndims;
        valueError(ss.str());
        return py::list();
    }
    if (shape[1] != Grid::NUM_MIDDLE_CELLS + NUM_GRID_CONFIGS) {
        std::stringstream ss;
        ss << "bits_and_configs has wrong shape in the last dimension: " << shape[1] << " . Expected "
                << Grid::NUM_MIDDLE_CELLS + NUM_GRID_CONFIGS;
        valueError(ss.str());
        return py::list();
    }
    const size_t batch_size = static_cast<size_t>(shape[0]);
    auto scales = vectorFromPyList<double>(py_scales);
    const shape4d_t out_shape{static_cast<npy_intp>(batch_size), 1, TAG_SIZE, TAG_SIZE};
    std::vector<std::vector<GeneratedGrid>> grids_vecs;
    buildGridsFromNpArr(arr_ptr, batch_size, grids_vecs);
    py::list grids;
    for(auto & scale : scales) {
        py::handle<> scaled_grids(drawGridsParallel(artist, grids_vecs, out_shape, scale));
        grids.append(scaled_grids);
    }
    return grids;
}


class PyGTDataLoader {
public:
    PyGTDataLoader(py::list pyfiles) :
            _gt_loader(vectorFromPyList<std::string>(pyfiles)) { };

    py::object batch(size_t batch_size, bool repeat) {
        GTRepeat repeat_enum;
        if (repeat) {
            repeat_enum = REPEAT;
        } else {
            repeat_enum = NO_REPEAT;
        }
        auto opt_batch = _gt_loader.batch(batch_size, repeat_enum);
        if(opt_batch) {
            auto & batch = opt_batch.get();
            std::cout << "imgs: " << batch.first.size() << ", grids: " << batch.second.size() << std::endl;
            auto pyarr_images = imagesToPyArray(batch.first, batch_size);
            auto pyarr_bits = gridToBitsPyArray(batch.second, batch_size);
            auto pyarr_config = gridToConfigPyArray(batch.second, batch_size);

            return py::make_tuple(
                    py::handle<>(pyarr_images),
                    py::handle<>(pyarr_bits),
                    py::handle<>(pyarr_config)
            );
        } else {
            return py::object(py::handle<>(Py_None));
        }
    };

private:
    PyObject * imagesToPyArray(std::vector<cv::Mat> images, size_t batch_size) {
        std::array<npy_intp, 4> images_shape{static_cast<npy_intp>(batch_size), 1, TAG_SIZE, TAG_SIZE};
        size_t image_count = get_count<4>(images_shape);
        uchar *raw_img_data = static_cast<uchar*>(malloc(image_count * sizeof(uchar)));
        uchar *iter_ptr = raw_img_data;
        for(auto & mat : images) {
            for(long j = 0; j < mat.rows; j++) {
                memcpy(iter_ptr, mat.ptr(j), mat.cols);
                iter_ptr += mat.cols;
            }
        }
        return newPyArrayOwnedByNumpy(images_shape, NPY_UBYTE, raw_img_data);
    }

    PyObject * gridToConfigPyArray(const std::vector<GroundTruthDatum> & grids, size_t batch_size) {
        const static size_t nb_configs = 6;
        std::array<npy_intp, 2> shape{static_cast<npy_intp>(batch_size), nb_configs};
        size_t count = get_count<2>(shape);
        std::cout << "[config] count: " << count * sizeof(float) << ", shp[0]: " << shape[0] << ", shp[1]: " << shape[1] << ", grids.size(): " << grids.size() << std::endl;
        float *raw_data = static_cast<float*>(malloc(count * sizeof(float)));
        float *ptr = raw_data;
        for(const auto & grid : grids) {
            *ptr = grid.z_rot; ++ptr;
            *ptr = grid.y_rot; ++ptr;
            *ptr = grid.x_rot; ++ptr;
            *ptr = grid.x; ++ptr;
            *ptr = grid.y; ++ptr;
            *ptr = grid.radius; ++ptr;
        }
        std::cout << "pointer diff: " << ptr - raw_data << std::endl;
        return newPyArrayOwnedByNumpy(shape, NPY_FLOAT32, raw_data);
    }

    PyObject * gridToBitsPyArray(const std::vector<GroundTruthDatum> & grids, size_t batch_size) {
        std::array<npy_intp, 2> shape{static_cast<npy_intp>(batch_size), Grid::NUM_MIDDLE_CELLS};
        const size_t count = get_count<2>(shape);
        std::cout << "[bits] count: " << count * sizeof(float) << ", shp[0]: " << shape[0] << ", shp[1]: " << shape[1] << ", grids.size(): " << grids.size() << std::endl;

        float *raw_data = static_cast<float*>(malloc(count * sizeof(float)));
        float *bits_ptr = raw_data;
        for(const auto & grid : grids) {
            memcpy(bits_ptr, &grid.bits[0], grid.bits.size()*sizeof(float));
            bits_ptr += grid.bits.size();
        }
        return newPyArrayOwnedByNumpy(shape, NPY_FLOAT32, raw_data);
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

BOOST_PYTHON_MODULE(pybeesgrid)
{
    init_numpy();
    py::def("generateBatch", &generateBatch);
    py::def("drawGrids", &drawGrids);

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
    py::scope().attr("NUM_MIDDLE_CELLS") = Grid::NUM_MIDDLE_CELLS;
    py::scope().attr("NUM_CONFIGS") = NUM_GRID_CONFIGS;

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

