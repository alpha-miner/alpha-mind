#ifndef pfopt_mv_optimizer_hpp
#define pfopt_mv_optimizer_hpp

#include "meanvariance.hpp"
#include <coin/IpIpoptApplication.hpp>

namespace pfopt {
    class PFOPT_CLASS MVOptimizer {
    public:
        MVOptimizer(const std::vector<double> &expectReturn,
            const std::vector<double> &varMatrix,
            const std::vector<double> &lbound,
            const std::vector<double> &ubound,
            double riskAversion);

        std::vector<double> xValue() const { return mvImpl_->xValue(); }
        double feval() const { return mvImpl_->feval(); }
        int status() const { return status_; }

    private:
        Ipopt::SmartPtr<MeanVariance> mvImpl_;
        Ipopt::SmartPtr<Ipopt::IpoptApplication> app_;
        Ipopt::ApplicationReturnStatus status_;
    };
}

#endif