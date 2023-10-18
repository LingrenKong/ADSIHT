#ifndef SRC_METRICS_H
#define SRC_METRICS_H

#include "Data.h"
#include "Algorithm.h"
#include <vector>
#include <random>
#include <algorithm>
// [[Rcpp::plugins("cpp11")]]

class Metric {
public:
  int ic_type;
  double ic_coef;
  Metric() = default;

  Metric(int ic_type, double ic_coef = 1.0) {
    this->ic_type = ic_type;
    this->ic_coef = ic_coef;
  };

  virtual double loss(Algorithm *algorithm, Data_Base &data) = 0;

  virtual double ic(Algorithm *algorithm, Data_Base &data) = 0;
};

class LmMetric : public Metric {
public:

  LmMetric(int ic_type, double ic_coef) : Metric(ic_type, ic_coef) {};

  double loss(Algorithm *algorithm, Data_Base &data);

  double ic(Algorithm *algorithm, Data_Base &data);
};

#endif //SRC_METRICS_H
