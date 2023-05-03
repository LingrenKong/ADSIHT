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

  virtual double loss(Algorithm *algorithm, Data &data) = 0;

  virtual double ic(Algorithm *algorithm, Data &data) = 0;
};

class LmMetric : public Metric {
public:

  LmMetric(int ic_type, double ic_coef) : Metric(ic_type, ic_coef) {};

  double loss(Algorithm *algorithm, Data &data) {
    return (data.y - data.x * algorithm->get_beta()).squaredNorm() / (data.n);
  }

  double ic(Algorithm *algorithm, Data &data) {
    if (ic_type == 0) {
      return this->loss(algorithm, data);
    } else if (ic_type == 1) {
      return double(data.n) * log(this->loss(algorithm, data)) +
        2.0 * algorithm->get_support_size();
    } else if (ic_type == 2) {
      return double(data.n) * log(this->loss(algorithm, data)) +
        log(double(data.n)) * algorithm->get_support_size();
    } else if (ic_type == 3) {
      return double(data.n) * log(this->loss(algorithm, data)) +
        this->ic_coef*log(double(data.p)) * log(log(double(data.n))) * algorithm->get_support_size();
    } else if (ic_type == 4) {
      return double(data.n) * log(this->loss(algorithm, data)) +
        (this->ic_coef*log(double(data.p)) +log(double(data.n))) * algorithm->get_support_size();
    } else return 0;
  }
};

#endif //SRC_METRICS_H
