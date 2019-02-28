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


class LinearRegression{
private:
    double min_error;
    uint max_iter;
    double learning_rate;
public:
    LinearRegression();
    ~LinearRegression(){};
    bool fit(const vector<vector<double>> &X, const vector<double> &y);
    vector<double> predict(const vector<vector<double>> &X);

    bool fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);
    VectorXd predict(const Matrix<double, Dynamic, Dynamic> &X);

    //vector<double> get_params();
    //score(X, y[, sample_weight]) 	Returns the coefficient of determination R^2 of the prediction.
    //set_params(**params) 	Set the parameters of this estimator.
    
    vector<double> coef_;
    double intercept_;
    VectorXd intercept_coef_;

    int error_flag_;
};


class Ridge{
private:
    double intercept_;
    VectorXd coef_; 
    int error_flag_;
public:
    double lr; 
    double alpha;
    uint max_iter;
    double tol;    
    bool normalize;

    Ridge(double alpha=0, double lr=0.1, uint max_iter=10000, double tol=1e-9, bool normalize=true);
    ~Ridge(){};

    bool valid(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);

    bool fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);
    VectorXd predict(const Matrix<double, Dynamic, Dynamic> &X);
    
    double get_intercept(){return intercept_;}
    VectorXd get_coef(){return coef_;}
};



class Lasso{
private:
    double intercept_;
    VectorXd coef_; 
    int error_flag_;
public:
    double lr; 
    double alpha;
    uint max_iter;
    double tol;    
    bool normalize;

    Lasso(double alpha=0, double lr=0.1, uint max_iter=10000, double tol=1e-9, bool normalize=true);
    ~Lasso(){};

    bool valid(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);

    bool fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);
    VectorXd predict(const Matrix<double, Dynamic, Dynamic> &X);
    
    double get_intercept(){return intercept_;}
    VectorXd get_coef(){return coef_;}
};







#endif