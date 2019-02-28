#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

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
    double min_error;
    uint max_iter;
    double learning_rate;
public:
    LogisticRegression();
    ~LogisticRegression(){};
    bool fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);
    VectorXd predict(const Matrix<double, Dynamic, Dynamic> &X);

    //bool fit(const vector<vector<double>> &X, const vector<double> &y);
    //vector<double> predict(const vector<vector<double>> &X);


    //vector<double> get_params();
    //score(X, y[, sample_weight]) 	Returns the coefficient of determination R^2 of the prediction.
    //set_params(**params) 	Set the parameters of this estimator.
    
    //vector<double> coef_;
    //double intercept_;
    VectorXd intercept_coef_;

    int error_flag_;
};




#endif