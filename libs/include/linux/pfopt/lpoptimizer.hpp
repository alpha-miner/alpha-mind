#ifndef pfopt_linear_programming_optimizer_hpp
#define pfopt_linear_programming_optimizer_hpp

#include "types.hpp"
#include "ClpSimplex.hpp"

namespace pfopt {

    class LpOptimizer {
    public:
        LpOptimizer(int numVariables,
                    int numCons,
                    double* constraintMatrix,
                    double* lowerBound,
                    double* upperBound,
                    double* objective);

        ~LpOptimizer() { delete [] sol_;}

        double* xValue() const { return sol_; }
        double feval() const;
        int status() const { return model_.status(); }

    private:
        ClpSimplex model_;
        size_t numberOfProb_;
        double* sol_;
    };
}

#endif