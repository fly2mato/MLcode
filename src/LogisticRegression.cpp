#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(double lr, uint max_iter, double tol, bool normalize) :
    lr(lr) ,
    max_iter(max_iter) ,
    tol(tol) ,
    normalize(normalize) {;}


VectorXd LogisticRegression::sigmoid(const VectorXd &z){
    return (1+(-z).array().exp()).cwiseInverse(); 
}


bool LogisticRegression::valid(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y){
    if (X.rows() != y.rows()) {
        return false;
    }
    return true;
}


bool LogisticRegression::fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y){
    if (!valid(X,y)) return false;

    Matrix<double, Dynamic, Dynamic> X_norm = Matrix<double, Dynamic, Dynamic>::Zero(X.rows(), X.cols()+1);
    X_norm.col(0) = MatrixXd::Ones(X.rows(), 1);

    VectorXd mean;
    VectorXd std;
    if (normalize) {
        mean = (X.colwise().sum())/X.rows();
        std = (X.array().pow(2).colwise().sum())/X.rows();
        X_norm.rightCols(X.cols()) = (X.array().rowwise() - mean.transpose().array()).rowwise() / std.transpose().array();
    } else {
        X_norm.rightCols(X.cols()) = X;
    }

    VectorXd intercept_coef_ = VectorXd::Zero(X.cols() + 1);    

    double cost;
    double cost_last = 0;

    VectorXd grad = VectorXd::Zero(X.cols()+1);
    uint iter = 0;

    while(1) {
        iter++;
        VectorXd s = sigmoid(X_norm * intercept_coef_);
        VectorXd H = s.array().log()*y.array() + (1-s.array()).log()*(1-y.array());
        cost = -H.sum();

        VectorXd grad = -(y - s).transpose()*X_norm;

        intercept_coef_ -= lr * grad;

        if (fabs(cost-cost_last) <= tol || iter > max_iter) break;  //stop condition
        cost_last = cost;
    }

    coef_ = intercept_coef_.bottomRows(X.cols());
    intercept_ = intercept_coef_(0);
    if (normalize) {
        coef_ = coef_.cwiseQuotient(std);
        intercept_ -= coef_.dot(mean);
    } 

    return true;
}

VectorXd LogisticRegression::predict(const Matrix<double, Dynamic, Dynamic> &X){
    if (X.cols() != coef_.rows()) {
        error_flag_ = 1;
        return VectorXd::Zero(1);
    }
    VectorXd z = (X * coef_).array() + intercept_;
    z = sigmoid(z);
    for(int i=0; i<z.rows(); ++i){
        if (z(i)>=0.5) z(i) = 1;
        else z(i)=0;
    }
    return z;
}


void LogisticRegression::read_data(ifstream & fid, int m, int n, Matrix<double, Dynamic, Dynamic> &X, VectorXd &y){
    for(int i=0; i<m; ++i){
        fid >> y(i);
        for(int j=0; j<n; ++j) {
            fid >> X(i,j);
        }
    }
}