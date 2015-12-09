
#include "beesgrid.h"

namespace beesgrid {

std::string getLabelsAsString(const Grid::idarray_t & id_arr) {
    std::stringstream ss;
    for (size_t i = 0; i < id_arr.size(); i++) {
        const auto & id = id_arr.at(i);
        if (i % 4 == 0 && i != 0) {
            ss << ".";
        }
        ss << id;
    }
    return ss.str();
}
}
