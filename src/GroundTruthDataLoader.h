#pragma once

#include "beesgrid.h"
#include <cereal/archives/json.hpp>
#include <biotracker/serialization/SerializationData.h>
#include <pipeline/util/GroundTruthEvaluator.h>
#include "Grid3D.h"
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

namespace beesgrid {

enum GTRepeat {
    REPEAT,
    NO_REPEAT,
};

template<typename Dtype>
class GroundTruthDataLoader {
public:
    GroundTruthDataLoader(const std::vector<std::string> &gt_files)
            : _gt_files(gt_files), _cache_idx(0), _gt_files_idx(0),
              _frame_index(0), _gt_first_file(true) {
    }

    boost::optional<dataset_t<Dtype>> batch(
            const size_t batch_size,
            const GTRepeat repeat = NO_REPEAT)
    {
        auto cache_left = [&]() { return cacheSize() - static_cast<long>(_cache_idx); };
        dataset_t<Dtype> dataset;
        auto & imgs = dataset.first;
        auto & labels = dataset.second;
        imgs.reserve(batch_size);
        labels.reserve(batch_size);
        while(cache_left() < batch_size) {
            if (! fillCache(repeat)) {
                break;
            }
        }
        if(_cache_idx + batch_size >= cacheSize()) {
            return boost::optional<dataset_t<Dtype>>();
        }
        const auto & img_begin = _img_cache.begin() + _cache_idx;
        const auto & labels_begin = _labels_cache.begin() + _cache_idx;
        imgs.insert(imgs.end(), img_begin, img_begin + batch_size);
        labels.insert(labels.end(), labels_begin, labels_begin + batch_size);
        _cache_idx += batch_size;
        maybeCleanCache();
        return dataset;
    }
protected:
    using gt_grid_vec_t = std::vector<GroundTruthGridSPtr>;
    using gt_dataset_t = std::pair<gt_grid_vec_t, cv::Mat>;
    boost::optional<gt_dataset_t> loadNextGTData(const GTRepeat repeat) {
        size_t n_frames = _gt_data.getFilenames().size();
        std::string image_path;
        gt_grid_vec_t grids;
        cv::Mat image;
        do {
            if (_frame_index < n_frames) {
                image_path = currentImageFileName();
            }
            if (!boost::filesystem::exists(image_path) ||
                    _frame_index >= n_frames ) {
                while (_frame_index >= n_frames ||
                       !boost::filesystem::exists(image_path)) {
                    if (_frame_index >= n_frames && !stepGTFilesIdx(repeat)) {
                        return boost::optional<gt_dataset_t>();
                    }
                    _gt_data = loadGTData(_gt_files.at(_gt_files_idx));
                    n_frames = _gt_data.getFilenames().size();
                    _frame_index = 0;
                    image_path = currentImageFileName();
                    if (!boost::filesystem::exists(image_path)) {
                        std::cerr << "[Warning] GT Image does not exists: "
                                 << image_path << std::endl;
                        _frame_index++;
                    }
                }
            }
            grids = getPipelineGridsForFrame(_gt_data, _frame_index);
            image = loadGTImage(image_path);
            _frame_index++;
        } while(grids.size() == 0);
        std::cout << "with " << grids.size() << " grids" << std::endl;
        return boost::make_optional<gt_dataset_t>(
                std::make_pair<gt_grid_vec_t , cv::Mat>(std::move(grids), std::move(image)));
    }
    std::string currentImageFileName() {
        auto base = _gt_data.getFilenames().at(_frame_index);
        boost::filesystem::path gt_file = _gt_files.at(_gt_files_idx);
        auto image_file = gt_file.remove_filename();
        image_file.append(base + ".jpeg");
        return boost::filesystem::absolute(image_file).string();
    }
    Serialization::Data loadGTData(const std::string &gt_path) {
        std::ifstream is(gt_path);
        assert(is.is_open());
        cereal::JSONInputArchive ar(is);
        Serialization::Data data;
        ar(data);
        return  data;
    }
    cv::Mat loadGTImage(const std::string &gt_path) {
        std::cout << "loading image: " << gt_path << std::endl;
        return cv::imread(gt_path, CV_LOAD_IMAGE_GRAYSCALE);
    }
    size_t cacheSize() {
        assert(_img_cache.size() == _labels_cache.size());
        return _img_cache.size();
    };
    bool stepGTFilesIdx(const GTRepeat repeat) {
        if (_gt_files_idx == 0 && _gt_first_file && _gt_files.size() > 0) {
            _gt_first_file = false;
            return true;
        }
        ++_gt_files_idx;
        if(_gt_files_idx >= _gt_files.size()) {
            if (repeat == REPEAT) {
                _gt_files_idx = 0;
            } else  {
                return false;
            }
        }
        return true;
    }

    void maybeCleanCache() {
        if (double(_cache_idx) / cacheSize() > 0.97) {
            size_t elems_to_copy = cacheSize() - static_cast<long>(_cache_idx);
            std::vector<cv::Mat> new_img_cache;
            new_img_cache.reserve(elems_to_copy);
            new_img_cache.insert(new_img_cache.end(),
                                 _img_cache.begin() + _cache_idx,
                                 _img_cache.end());
            std::vector<std::vector<Dtype>> new_labels_cache;
            new_labels_cache.reserve(elems_to_copy);
            new_labels_cache.insert(new_labels_cache.end(),
                                 _labels_cache.begin() + _cache_idx,
                                 _labels_cache.end());
            _img_cache = std::move(new_img_cache);
            _labels_cache = std::move(new_labels_cache);
            _cache_idx = 0;
        }
    }

    bool fillCache(const GTRepeat repeat) {
        const auto opt_gt_dataset = loadNextGTData(repeat);
        if (! opt_gt_dataset) { return false; }

        const auto & gt_grids = opt_gt_dataset.get().first;
        const auto & image = opt_gt_dataset.get().second;
        for(const auto & gt_grid : gt_grids) {
            const auto center = gt_grid->getCenter();
            cv::Rect box{center.x - int(TAG_SIZE / 2), center.y - int(TAG_SIZE / 2),
                         TAG_SIZE, TAG_SIZE};
            if(box.x > 0 && box.y > 0 && box.x + box.width < image.cols && box.y + box.height < image.rows) {
                // TODO:: add border
                _img_cache.emplace_back(image(box).clone());
                std::vector<float> bits;
                for(const auto & bit : triboolIDtoVector<Dtype>(gt_grid->getIdArray())) {
                    bits.push_back(bit);
                }
                _labels_cache.push_back(bits);
            }
        }
        return true;
    }
    Serialization::Data _gt_data;
    std::vector<std::string> _gt_files;
    std::vector<cv::Mat> _img_cache;
    std::vector<std::vector<Dtype>> _labels_cache;
    size_t _cache_idx;
    size_t _gt_files_idx;
    size_t _frame_index;
    bool _gt_first_file;
};
}


