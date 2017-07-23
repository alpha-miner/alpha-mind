#ifndef pfopt_mean_variance_hpp
#define pfopt_mean_variance_hpp

#include "types.hpp"
#include <vector>
#include <coin/IpTNLP.hpp>

using Ipopt::TNLP;
using Ipopt::Number;
using Ipopt::Index;
using Ipopt::SolverReturn;
using Ipopt::IpoptData;
using Ipopt::IpoptCalculatedQuantities;

namespace pfopt {

	class PFOPT_CLASS MeanVariance : public TNLP {

    public:
        MeanVariance(const std::vector<double> &expectReturn,
                     const std::vector<double> &varMatrix,
                     double riskAversion=1.);

        bool setBoundedConstraint(const std::vector<double>& lb, const std::vector<double>& ub);

        virtual bool get_nlp_info(Index &n, Index &m, Index &nnz_jac_g,
                                  Index &nnz_h_lag, IndexStyleEnum &index_style);

        virtual bool get_bounds_info(Index n, Number *x_l, Number *x_u,
                                     Index m, Number *g_l, Number *g_u);

        virtual bool get_starting_point(Index n, bool init_x, Number *x,
                                        bool init_z, Number *z_L, Number *z_U,
                                        Index m, bool init_lambda,
                                        Number *lambda);

        virtual bool eval_f(Index n, const Number *x, bool new_x, Number &obj_value);

        virtual bool eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f);

        virtual bool eval_g(Index n, const Number *x, bool new_x, Index m, Number *g);

        virtual bool eval_jac_g(Index n, const Number *x, bool new_x,
                                Index m, Index nele_jac, Index *iRow, Index *jCol,
                                Number *values);

        virtual bool eval_h(Index n, const Number* x, bool new_x,
                            Number obj_factor, Index m, const Number* lambda,
                            bool new_lambda, Index nele_hess, Index* iRow,
                            Index* jCol, Number* values);

        virtual void finalize_solution(SolverReturn status,
                                       Index n, const Number *x, const Number *z_L, const Number *z_U,
                                       Index m, const Number *g, const Number *lambda,
                                       Number obj_value,
                                       const IpoptData *ip_data,
                                       IpoptCalculatedQuantities *ip_cq);

        double feval() const { return feval_; }
        std::vector<double> xValue() const { return x_; }

    private:
        VectorXd expectReturn_;
        MatrixXd varMatrix_;
        int numOfAssets_;
        VectorXd xReal_;
        double riskAversion_;

        VectorXd lb_;
        VectorXd ub_;
        VectorXd grad_f_;
        double feval_;
        std::vector<double> x_;
    };
}
#endif
