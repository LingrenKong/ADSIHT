#ifndef SRC_DATA_H
#define SRC_DATA_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include "normalize.h"
// [[Rcpp::plugins("cpp11")]]
using namespace std;

class Data_Base {
public:
  Eigen::MatrixXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd weight;
  Eigen::VectorXd x_mean;
  Eigen::VectorXd x_norm;
  double y_mean;
  int n;
  int p;
  int g_num;
  Eigen::VectorXi g_index;
  Eigen::VectorXi g_size;

  Data_Base() = default;//no construction, only for data-type define

  void add_weight();

  Eigen::VectorXi get_g_index();

  Eigen::VectorXi get_g_size();
};

class Data: public Data_Base{
public:
  Data(Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd& weight, Eigen::VectorXi& g_index);
  //construction function with normalization

  void normalize();
};

class Data_logit: public Data_Base{
public:
  Data_logit(Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd& weight, Eigen::VectorXi& g_index);
  //construction function with normalization only on x

  void normalize_x();
};
#endif //SRC_DATA_H
