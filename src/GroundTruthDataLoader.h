#pragma once

#include "deepdecoder.h"
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/GroundTruthEvaluator.h>

#include <cereal/archives/json.hpp>
#include <biotracker/tracking/serialization/SerializationData.h>
#include <biotracker/tracking/algorithm/BeesBook/ImgAnalysisTracker/legacy/Grid3D.h>

namespace deepdecoder {

template<typename Dtype>
class GroundTruthDataLoader {
public:
    GroundTruthDataLoader(const std::vector<std::string> &gt_files)
            : _gt_files(gt_files), _cache_idx(0), _gt_files_idx(0) {
    }

    boost::optional<dataset_t<Dtype>> batch(size_t batch_size, bool repeat = false) {
        auto cache_left = [&]() { return cacheSize() - static_cast<long>(_cache_idx); };
        dataset_t<Dtype> dataset;
        auto & imgs = dataset.first;
        auto & labels = dataset.second;
        size_t step = batch_size;
        while(cache_left() < batch_size) {
            if (! fillCache(repeat)) {
                step = cache_left();
                break;
            }
        }
        if(_cache_idx >= cacheSize()) {
            return boost::optional<dataset_t<Dtype>>();
        }
        const auto & img_begin = _img_cache.begin() + _cache_idx;
        const auto & labels_begin = _labels_cache.begin() + _cache_idx;
        imgs.insert(imgs.end(), img_begin, img_begin + step);
        labels.insert(labels.end(), labels_begin, labels_begin + step);
        _cache_idx += step;
        maybeCleanCache();
        return dataset;
    }
protected:
    std::vector<GroundTruthGridSPtr> loadGTData(const std::string &gt_path) {
        std::ifstream is(gt_path + ".tdat");
        cereal::JSONInputArchive ar(is);
        Serialization::Data data;
        ar(data);
        return getPipelineGridsForFrame(data, 0);
    }
    cv::Mat loadGTImage(const std::string &gt_path) {
        return cv::imread(gt_path + ".jpeg", CV_LOAD_IMAGE_GRAYSCALE);
    }
    size_t cacheSize() {
        CHECK_EQ(_img_cache.size(), _labels_cache.size());
        return _img_cache.size();
    };
    bool stepGTFilesIdx(bool repeat) {
        ++_gt_files_idx;
        if(_gt_files_idx >= _gt_files.size()) {
            if (repeat) {
                _gt_files_idx = 0;
            } else  {
                return false;
            }
        }
        return true;
    }

    void maybeCleanCache() {
        if (double(_cache_idx) / cacheSize() > 0.95) {
            size_t elems_to_copy = cacheSize() - static_cast<long>(_cache_idx);
            std::vector<cv::Mat> new_img_cache;
            new_img_cache.reserve(elems_to_copy);
            new_img_cache.insert(new_img_cache.end(),
                                 _img_cache.begin() + _cache_idx,
                                 _img_cache.end());
            std::vector<std::vector<Dtype>> new_labels_cache;
            new_labels_cache.reserve(elems_to_copy);
            new_img_cache.insert(new_img_cache.end(),
                                 _img_cache.begin() + _cache_idx,
                                 _img_cache.end());
            _cache_idx = 0;
        }
    }

    bool fillCache(bool repeat) {
        if (_gt_files_idx >= _gt_files.size()) { return false; }
        const auto gt_grids = loadGTData(_gt_files.at(_gt_files_idx));
        const auto image = loadGTImage(_gt_files.at(_gt_files_idx));
        stepGTFilesIdx(repeat);
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

    std::vector<std::string> _gt_files;
    std::vector<cv::Mat> _img_cache;
    std::vector<std::vector<Dtype>> _labels_cache;
    size_t _cache_idx;
    size_t _gt_files_idx;
};
}


