#ifndef pfopt_qp_alglib_hpp
#define PFOPT_qp_alglib_HPP

#include "types.hpp"
#include <optimization.h>

using namespace alglib;

namespace pfopt {

    class PFOPT_CLASS AlglibData {
    public:
        AlglibData(const std::vector<double> &expectReturn,
                   const std::vector<double> &varMatrix,
                   double riskAversion=1.);

        real_1d_array b() const;
        real_2d_array a() const;
        real_1d_array x0() const;
        real_1d_array bndl() const;
        real_1d_array bndu() const;
        real_1d_array scale() const;
        real_2d_array c() const;
        integer_1d_array ct() const;

    private:
        real_1d_array b_;
        real_2d_array a_;
        size_t n_;
    };

}

#endif
