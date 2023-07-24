#include "Metric.h"

double LmMetric::loss(Algorithm *algorithm, Data &data) {
  return (data.y - data.x * algorithm->get_beta()).squaredNorm() / (data.n);
}

double LmMetric::ic(Algorithm *algorithm, Data &data) {
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
