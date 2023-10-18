#include "Algorithm.h"
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

void DSIHTLm::fit(double s_0, double ic_coef) {
  Data_Base data = this->data;
  Eigen::MatrixXd x = data.x;
  Eigen::VectorXd y = data.y;
  int n = data.n;
  int p = data.p;
  int m = data.g_num;
  Eigen::VectorXi gindex = data.get_g_index();
  Eigen::VectorXi gsize = data.get_g_size();
  int d = gsize.maxCoeff();
  double rho = this->rho;
  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
  double delta = Delta(s_0, m, d);
  double lam1, lam0 = pow(rho, ceil(s_0/3)-1)*max(std::sqrt(delta*y.squaredNorm()/n/n), ((x.transpose()*y/n).cwiseAbs()).maxCoeff());
  // Rcout<<"----------------------\n";
  while(1) {
    beta1 = tau(x, y, beta0, gindex, gsize, lam0, s_0, m, p);
    lam1 = rho*rho*lam0;
    if (lam1 >= 2*sqrt((y-x*beta1).squaredNorm()/n*delta/n)) {
      beta0 = beta1;
      lam0 = lam1;
    } else {
      break;
    }
  }
  beta1 = least_square(x, y, beta1, p);
  double delta_tbar = sqrt((y-x*beta1).squaredNorm()/n);
  double ic0, ic1 = IC(x, y, beta1, gindex, gsize, s_0, n, m, p, d, delta_tbar, ic_coef);
  this->ic = ic1;
  this->beta = beta1;
  this->lam = lam1;
  while (1) {
    beta0 = tau(x, y, beta1, gindex, gsize, lam1, s_0, m, p);
    beta0 = least_square(x, y, beta0, p);
    // Rcout<<"size:"<<(beta0.array() != 0).count()<<"\n";
    if ((beta0.array() != 0).count() >= p/5 || (beta0.array() != 0).count() >= x.rows()) break;
    lam0 = rho*lam1;
    ic0 = IC(x, y, beta0, gindex, gsize, s_0, n, m, p, d, delta_tbar, ic_coef);
    // Rcout<<"-IC0:"<<ic0<<"\n";
    // Rcout<<"-IC1:"<<this->ic<<"\n";
    if (ic0 < this->ic) {
      this->beta = beta0;
      this->lam = lam0;
      this->ic = ic0;
      beta1 = beta0;
      // Rcout<<"Update"<<"\n";
    }
    lam1 = lam0;
    if (lam1 < log(exp(1)*d/s_0)*delta_tbar/n) break;
  }
  this->size = (this->beta.array() != 0).count();
  this->A_out = support_set(this->beta, p, this->size)+Eigen::VectorXi::Ones(this->size);
}

void DSIHT_logit::fit(double s_0, double ic_coef) {
  Data_Base data = this->data;
  Eigen::MatrixXd x = data.x;
  Eigen::VectorXd y = data.y;
  int n = data.n;
  int p = data.p;
  int m = data.g_num;
  Eigen::VectorXi gindex = data.get_g_index();
  Eigen::VectorXi gsize = data.get_g_size();
  int d = gsize.maxCoeff();
  double rho = this->rho;
  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
  double delta = Delta(s_0, m, d);
  double lam1, lam0 = pow(rho, ceil(s_0/3)-1)*max(std::sqrt(delta*y.squaredNorm()/n/n), ((x.transpose()*y/n).cwiseAbs()).maxCoeff());
  // Rcout<<"----------------------\n";
  while(1) {
    beta1 = tau_logit(x, y, beta0, gindex, gsize, lam0, s_0, m, p);
    lam1 = rho*rho*lam0;
    if (lam1 >= 2*sqrt((y-x*beta1).squaredNorm()/n*delta/n)) {
      beta0 = beta1;
      lam0 = lam1;
    } else {
      break;
    }
  }
  //beta1 = IWLS(x, y, beta1, p);
  double delta_tbar = sqrt((y-x*beta1).squaredNorm()/n);
  double ic0, ic1 = IC(x, y, beta1, gindex, gsize, s_0, n, m, p, d, delta_tbar, ic_coef);
  this->ic = ic1;
  this->beta = beta1;
  this->lam = lam1;
  while (1) {
    beta0 = tau_logit(x, y, beta1, gindex, gsize, lam1, s_0, m, p);
    //beta0 = IWLS(x, y, beta0, p);
    Rcout<<"size:"<<(beta0.array() != 0).count()<<"\n";
    if ((beta0.array() != 0).count() >= p/5 || (beta0.array() != 0).count() >= x.rows()) break;
    lam0 = rho*lam1;
    ic0 = IC(x, y, beta0, gindex, gsize, s_0, n, m, p, d, delta_tbar, ic_coef);
    Rcout<<"-IC0:"<<ic0<<"\n";
    Rcout<<"-IC1:"<<this->ic<<"\n";
    if (ic0 < this->ic) {
      this->beta = beta0;
      this->lam = lam0;
      this->ic = ic0;
      beta1 = beta0;
      Rcout<<"Update"<<"\n";
    }
    lam1 = lam0;
    if (lam1 < log(exp(1)*d/s_0)*delta_tbar/n) break;
  }
  this->size = (this->beta.array() != 0).count();
  this->A_out = support_set(this->beta, p, this->size)+Eigen::VectorXi::Ones(this->size);
}
