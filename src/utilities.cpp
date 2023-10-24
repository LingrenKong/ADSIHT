#include "utilities.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace std;

double Delta(double s_0, int m, int d) {
  return (1/s_0*log(exp(1)*m)+log(exp(1)*d/s_0));
}

Eigen::VectorXd tau(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXi &gindex, Eigen::VectorXi &gsize, double lambda, double s_0, int m, int p) {
    Eigen::VectorXd temp = beta+X.transpose()*(y-X*beta)/X.rows();
     for (int i = 0; i < p; i++) {
        if (abs(temp(i)) < lambda) temp(i) = 0.0;
     }
     for (int i = 0; i < m; i++) {
        if (temp.segment(gindex(i), gsize(i)).squaredNorm() < s_0*pow(lambda, 2)) {
          temp.segment(gindex(i), gsize(i)) = Eigen::VectorXd::Zero(gsize(i));
        }
     }
     return temp;
}

double logit_transform(double x) // the functor we want to apply
{
  if (x > 1e10) return 1-1e-7;
  if (x < 1e-10) return 1e-7;
  return 1 / (1 + exp(-x));
}

Eigen::VectorXd logit_b1(Eigen::VectorXd theta){
  return theta.unaryExpr(&logit_transform);
}

Eigen::VectorXd tau_logit(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXi &gindex, Eigen::VectorXi &gsize, double lambda, double s_0, int m, int p) {
  Eigen::VectorXd temp = beta+X.transpose()*(y-logit_b1(X*beta))/X.rows();
  for (int i = 0; i < p; i++) {
    if (abs(temp(i)) < lambda) temp(i) = 0.0;
  }
  for (int i = 0; i < m; i++) {
    if (temp.segment(gindex(i), gsize(i)).squaredNorm() < s_0*pow(lambda, 2)) {
      temp.segment(gindex(i), gsize(i)) = Eigen::VectorXd::Zero(gsize(i));
    }
  }
  return temp;
}


int group_support_size(Eigen::VectorXd &beta, Eigen::VectorXi &gindex, Eigen::VectorXi &gsize, int m) {
  int size = 0;
  Eigen::VectorXd temp = Eigen::VectorXd::Zero(m);
  for (int i = 0; i < m; i++) {
    temp(i) = beta.segment(gindex(i), gsize(i)).squaredNorm();
    if (temp(i) != 0 ) {
        size = size + 1;
    }
  }
  return size;
}

Eigen::VectorXi support_set(Eigen::VectorXd &beta, int p, int size) {
  Eigen::VectorXi temp = Eigen::VectorXi::Zero(size);
  int flag = 0;
  for (int i = 0; i < p; i++) {
    if (beta(i) != 0) {
      temp(flag) = i;
      flag = flag+1;
    }
  }
  return temp;
}

Eigen::VectorXd least_square(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, int p) {
  int size = (beta.array() != 0).count();
  if (size >= X.rows() || size == 0) {
    return beta;
  }
  else{
  Eigen::VectorXi set = support_set(beta, p, size);
  Eigen::MatrixXd X_temp = Eigen::MatrixXd::Zero(X.rows(), size);
  for (int i = 0; i < size; i++) {
    X_temp.col(i) = X.col(set(i));
  }
  Eigen::VectorXd temp = X_temp.colPivHouseholderQr().solve(y);
  Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(p);
  for (int i = 0; i < size; i++) {
    beta_hat(set(i)) = temp(i);
  }
  return(beta_hat);
  }
}

// [[Rcpp::export]]
Eigen::VectorXd IWLS(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, int p) {
  //Rcout << "running IWLS" << "\n";
  int size = (beta.array() != 0).count();
  if (size >= X.rows() || size == 0) {
    return beta;
  }
  else{
    Eigen::VectorXi set = support_set(beta, p, size);
    //Rcout << "set:\n" << set << "\n";
    Eigen::MatrixXd X_temp = Eigen::MatrixXd::Zero(X.rows(), size);
    for (int i = 0; i < size; i++) {
      X_temp.col(i) = X.col(set(i));
    }
    Eigen::MatrixXd beta_cut = Eigen::VectorXd::Zero(size);
    for (int i = 0; i < size; i++) {
      beta_cut(i) = beta(set(i));
    }

    Rcpp::Environment stats("package:stats");
    Rcpp::Function glm_fit = stats["glm.fit"];
    //Rcout << "glm_fit:\n" << glm_fit << "\n";
    Rcpp::Function binomial = stats["binomial"];
    List result = glm_fit(X_temp,y,Named("family",binomial(Named("link","logit"))));
    Rcpp::NumericVector glm_coef = result[0];
    //beta_cut = result[0];
    //Rcout << "result:\n" << result << "\n";
    //Rcout << "glm_coef:\n" << glm_coef << "\n";
    beta_cut = as<Eigen::VectorXd>(glm_coef);
    // //Rcout << "beta_cut:\n" << beta_cut << "\n";
    // Eigen::MatrixXd beta_tmp = beta_cut;
    // Eigen::VectorXd eta,diff_tmp,Z,mu,v;Eigen::MatrixXd W;
    // Rcout << "running iteration" << "\n";
    // for(int iter = 0; iter < 1000; iter++) {//maxiter=100
    //   eta = X_temp*beta_cut;
    //   beta_tmp = beta_cut;
    //   mu = logit_b1(eta);
    //   //Rcout << "mu:\n" << mu << "\n";
    //   v = mu*(Eigen::VectorXd::Ones(mu.size())-mu);
    //   W = v.asDiagonal();
    //   //Rcout << "W:\n" << W << "\n";
    //   Z = eta + ( (y-mu).array()/v.array() ).matrix();
    //   //Rcout << "Z:\n" << Z << "\n";
    //   beta_cut = (X_temp.transpose()*W*X_temp).inverse()*X_temp.transpose()*W*Z;
    //   //Rcout << "beta_cut:\n" << beta_cut << "\n";
    //   diff_tmp = beta_tmp-beta_cut;
    //   double D = diff_tmp.array().abs().maxCoeff();
    //   if(D<1e-8){
    //     Rcout << "break in iter" << iter << "\n";
    //     break;
    //   }
    // }

    Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(p);
    for (int i = 0; i < size; i++) {
      beta_hat(set(i)) = beta_cut(i);
    }
    return(beta_hat);
  }
}

double IC(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXi &gindex, Eigen::VectorXi &gsize, double s_0, int n, int m, int p, int d, double delta_t, double ic_coef) {
  int size1 = group_support_size(beta, gindex, gsize, m);
  int size2 = (beta.array() != 0).count();
  double size3;
  if (size1 > size2/s_0) {
    size3 = size1;
  }
  else {
    size3 = size2/s_0;
  }
  double omega = size3*log(exp(1)*m/size3)+s_0*size3*log(exp(1)*d/s_0);
  double ic = (y-X*beta).squaredNorm()/n+ic_coef*0.5*omega*pow(delta_t, 2)/n;
  if (isnan(ic)) {
    return 1e10;
  }
  else {
    return ic;
  }
}

double IC_logit(Eigen::MatrixXd &X, Eigen::VectorXd &y, Eigen::VectorXd &beta, Eigen::VectorXi &gindex, Eigen::VectorXi &gsize, double s_0, int n, int m, int p, int d, double delta_t, double ic_coef) {
  int size1 = group_support_size(beta, gindex, gsize, m);
  int size2 = (beta.array() != 0).count();
  double size3;
  if (size1 > size2/s_0) {
    size3 = size1;
  }
  else {
    size3 = size2/s_0;
  }
  double omega = size3*log(exp(1)*m/size3)+s_0*size3*log(exp(1)*d/s_0);
  Eigen::VectorXd pp = logit_b1(X*beta);
  double ic = (-y.array()*pp.array().log()-(1-y.array())*(1-pp.array()).log()).sum() / n +ic_coef*0.5*omega*pow(delta_t, 2)/n;
  if (isnan(ic)) {
    return 1e10;
  }
  else {
    return ic;
  }
}
