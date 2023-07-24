#ifndef SRC_NORMALIZE_H
#define SRC_NORMALIZE_H

#include <RcppEigen.h>

void Normalize(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &weights, Eigen::VectorXd &meanx, double &meany, Eigen::VectorXd &normx);

#endif //SRC_NORMALIZE_H
