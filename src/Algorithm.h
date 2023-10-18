#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H
#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include "utilities.h"
#include "Data.h"
#include "math.h"
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]
using namespace std;

class Algorithm {
public:
  Data_Base data;
  Eigen::VectorXd beta;
  Eigen::VectorXi A_out;
  int size;
  double lam;
  double rho;
  double ic;

  Algorithm() = default;

  Algorithm(Data_Base &data)
  {
    this->data = data;
    this->beta = Eigen::VectorXd::Zero(data.p);
    this->lam = 0.0;
    this->ic = 0.0;
    this->size = 0;
  };

  void update_rho(double rho) {
    this->rho = rho;
  }

  double get_lambda() {
    return this->lam;
  }

  double get_ic() {
    return this->ic;
  }

  Eigen::VectorXd get_beta() {
    return this->beta;
  }

  int get_support_size() {
    return this->size;
  }

  Eigen::VectorXi get_A_out() {
    return this->A_out;
  }

  virtual void fit(double s_0, double ic_coef)=0;
};

class DSIHTLm : public Algorithm {
public:
  DSIHTLm(Data_Base &data) : Algorithm(data){};
  void fit(double s_0, double ic_coef);
};

class DSIHT_logit : public Algorithm {
public:
  DSIHT_logit(Data_Base &data) : Algorithm(data){};
  void fit(double s_0, double ic_coef);
};

#endif //SRC_ALGORITHM_H
