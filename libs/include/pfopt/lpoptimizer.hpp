#ifndef pfopt_linear_programming_optimizer_hpp
#define pfopt_linear_programming_optimizer_hpp

#include "types.hpp"
#include <vector>
#include "ClpSimplex.hpp"

namespace pfopt {

    class PFOPT_CLASS LpOptimizer {
        public:
            LpOptimizer(const std::vector<double>& constraintsMatraix,
                        const std::vector<double>& lowerBound,
                        const std::vector<double>& upperBound,
                        const std::vector<double>& objective);

            std::vector<double> xValue() const;
			double feval() const;
            int status() const { return model_.status();}

        private:
            ClpSimplex model_;
			int numberOfProb_;
    };
}

#endif