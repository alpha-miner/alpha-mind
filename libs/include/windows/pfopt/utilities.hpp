#ifndef pfopt_utilities_hpp
#define pfopt_utilities_hpp

#include "types.hpp"

namespace pfopt {

    namespace io {
        std::vector<double> read_csv(const std::string &filePath);
    }

    double min(const real_1d_array& array, int n = 0);
    double max(const real_1d_array& array, int n = 0);
    double sum(const real_1d_array& array, int n = 0);

    bool is_close(double a, double b, double tol=1e-9);
}

#endif