#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <list>
#include <algorithm>
#include <string>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;


#define EPSILON 1e-6


class LogisticRegression{
private:
    double intercept_;
    VectorXd coef_; 
    int error_flag_;
public:
    double lr; 
    uint max_iter;
    double tol;    
    bool normalize;

    LogisticRegression(double lr=0.1, uint max_iter=10000, double tol=1e-9, bool normalize=true);
    ~LogisticRegression(){};
    bool fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);
    VectorXd predict(const Matrix<double, Dynamic, Dynamic> &X);

    VectorXd sigmoid(const VectorXd &z);
    double get_intercept(){return intercept_;}
    VectorXd get_coef(){return coef_;}

    bool valid(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);
    void read_data(ifstream & fid, int m, int n, Matrix<double, Dynamic, Dynamic> &X, VectorXd &y);
};




#endif