// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// DSIHT_Cpp
List DSIHT_Cpp(Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd& weight, int ic_type, double ic_scale, Eigen::VectorXd& sequence, double kappa, Eigen::VectorXi& g_index, double ic_coef, bool method, double coef1, double coef2, double eta, int max_iter);
RcppExport SEXP _ADSIHT_DSIHT_Cpp(SEXP xSEXP, SEXP ySEXP, SEXP weightSEXP, SEXP ic_typeSEXP, SEXP ic_scaleSEXP, SEXP sequenceSEXP, SEXP kappaSEXP, SEXP g_indexSEXP, SEXP ic_coefSEXP, SEXP methodSEXP, SEXP coef1SEXP, SEXP coef2SEXP, SEXP etaSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< int >::type ic_type(ic_typeSEXP);
    Rcpp::traits::input_parameter< double >::type ic_scale(ic_scaleSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type sequence(sequenceSEXP);
    Rcpp::traits::input_parameter< double >::type kappa(kappaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXi& >::type g_index(g_indexSEXP);
    Rcpp::traits::input_parameter< double >::type ic_coef(ic_coefSEXP);
    Rcpp::traits::input_parameter< bool >::type method(methodSEXP);
    Rcpp::traits::input_parameter< double >::type coef1(coef1SEXP);
    Rcpp::traits::input_parameter< double >::type coef2(coef2SEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(DSIHT_Cpp(x, y, weight, ic_type, ic_scale, sequence, kappa, g_index, ic_coef, method, coef1, coef2, eta, max_iter));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ADSIHT_DSIHT_Cpp", (DL_FUNC) &_ADSIHT_DSIHT_Cpp, 14},
    {NULL, NULL, 0}
};

RcppExport void R_init_ADSIHT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
