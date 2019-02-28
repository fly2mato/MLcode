#include "LinearRegression.h"

LinearRegression::LinearRegression() :
    min_error(1e-10),
    max_iter(10000),
    learning_rate(0.1),
    error_flag_(-1) {}

vector<double> LinearRegression::predict(const vector<vector<double>> &X){
    if (!X.size() || X[0].size() != coef_.size()) {
        error_flag_ = 1;
        return {};
    }

    vector<double> ans;
    double predict;    
    
    
    ans.clear();
    for(uint i=0; i<X.size(); ++i){
        predict = intercept_;
        for(uint j=0; j<X[0].size(); ++j) {
            predict += coef_[j] * X[i][j];
        }
        ans.push_back(predict);
    }

    return ans;
}


VectorXd LinearRegression::predict(const Matrix<double, Dynamic, Dynamic> &X){
    if (X.cols() != intercept_coef_.rows()-1) {
        error_flag_ = 1;
        return VectorXd::Zero(1);
    }

    VectorXd pre = (X * intercept_coef_.bottomRows(intercept_coef_.rows()-1)).array() + intercept_coef_(0);
    return pre;
}


bool LinearRegression::fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y){
    if (X.rows() != y.rows()) {
        error_flag_ = 1;
        return false;
    }

    VectorXd mean = (X.colwise().sum())/X.rows();
    VectorXd std = (X.array().pow(2).colwise().sum())/X.rows();

    Matrix<double, Dynamic, Dynamic> X_norm = Matrix<double, Dynamic, Dynamic>::Zero(X.rows(), X.cols()+1);
    X_norm.col(0) = MatrixXd::Ones(X.rows(), 1);
    X_norm.rightCols(X.cols()) = (X.array().rowwise() - mean.transpose().array()).rowwise() / std.transpose().array();

    intercept_coef_ = VectorXd::Zero(X.cols() + 1);    
    //intercept_coef_(0) = 1;

    double cost;
    double cost_last = 0;

    VectorXd grad = VectorXd::Zero(X.cols()+1);
    uint iter = 0;

    while(1) {
        iter++;
        cost = 0;

        VectorXd y_pred = X_norm * intercept_coef_;
        VectorXd error = y - y_pred;
        
        cost = error.dot(error)/X_norm.rows();
        grad = -2.0/X_norm.rows() * X_norm.transpose() * error;

        intercept_coef_ -= learning_rate * grad;

        if (fabs(cost-cost_last) <= min_error || iter > max_iter) break;  //stop condition
        cost_last = cost;
    }

    //invert normlization
    intercept_coef_.bottomRows(X.cols()) =  intercept_coef_.bottomRows(X.cols()).cwiseQuotient(std);
    intercept_coef_(0) -= intercept_coef_.bottomRows(X.cols()).dot(mean);
    
    return true;
}





bool LinearRegression::fit(const vector<vector<double>> &X, const vector<double> &y){
    if (X.size() != y.size()) {
        error_flag_ = 1;
        return false;
    }
    for(auto i : X) {
        if (i.size() != X[0].size()) {
            error_flag_ = 2;
            return false;
        }
    }


    //============ normlization  x = (X-mu)/std
    vector<double> mean(X[0].size(), 0);
    vector<double> std(X[0].size(), 0);
    double x_2, x;
    for(uint j=0; j<X[0].size(); ++j) {
        x_2 = x = 0;
        for(uint i=0; i<X.size(); ++i){
            x_2 += X[i][j]*X[i][j];
            x += X[i][j];
        }
        mean[j] = x / X.size();
        std[j] = EPSILON + sqrt(x_2/X.size() - mean[j]*mean[j]);
    }
    //============



    intercept_ = 0.0;
    coef_.clear();
    for(uint i = 0; i < X[0].size(); ++i) coef_.push_back(0);


    double cost;
    double cost_last = 0;
    double loss;
    double error;
    vector<double> gradient(X[0].size()+1, 0);
    uint iter = 0;
    while(1) {
        iter++;
        cost = 0;
        for(uint j=0; j<X[0].size()+1; ++j) gradient[j] = 0;

        for(uint i=0; i<y.size(); ++i) {
            error = y[i];
            for(uint j=0; j<X[0].size(); ++j){
                error -= coef_[j] * (X[i][j] - mean[j])/std[j];
            }
            error -= intercept_;
            loss = error*error;
            cost += loss;

            for(uint j=0; j<X[0].size(); ++j){
                gradient[j] -= error * (X[i][j]-mean[j])/std[j] * 2/y.size();
            }
            gradient.back() -= error * 2/y.size();
        }
        cost /= y.size();

        intercept_ -= learning_rate * gradient.back();
        for(uint j=0; j<X[0].size(); ++j){
            coef_[j] -= learning_rate * gradient[j];
        }


        if (fabs(cost-cost_last) <= min_error || iter > max_iter) break;  //stop condition
        cost_last = cost;
    }

    //invert normlization
    for(uint j=0; j<X[0].size(); ++j){
        intercept_ -= (coef_[j] * mean[j])/std[j];
        coef_[j] /= std[j];
    }

    return true;
}






//====================Ridge
Ridge::Ridge(double alpha, double lr, uint max_iter, double tol, bool normalize) :
lr(lr) ,
alpha(alpha) ,
max_iter(max_iter) ,
tol(tol) ,
normalize(normalize) {;}



bool Ridge::valid(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y){
    if (X.rows() != y.rows()) {
        return false;
    }
    return true;
}

bool Ridge::fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y){
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
    VectorXd intercept_coef_2 = VectorXd::Zero(X.cols() + 1);    

    double cost;
    double cost_last = 0;

    VectorXd grad = VectorXd::Zero(X.cols()+1);
    uint iter = 0;

    while(1) {
        iter++;
        cost = 0;

        VectorXd y_pred = X_norm * intercept_coef_;
        VectorXd error = y - y_pred;
        
        cost = error.dot(error)/X_norm.rows();

        // intercept_coef_2 = intercept_coef_;
        // intercept_coef_2(0) = 0;
        grad = -2.0/X_norm.rows() * X_norm.transpose() * error + 2.0 * alpha * intercept_coef_;

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

VectorXd Ridge::predict(const Matrix<double, Dynamic, Dynamic> &X){
    if (X.cols() != coef_.rows()) {
        error_flag_ = 1;
        return VectorXd::Zero(1);
    }
    VectorXd pre = (X * coef_).array() + intercept_;
    return pre;
}






//====================Lasso
Lasso::Lasso(double alpha, double lr, uint max_iter, double tol, bool normalize) :
lr(lr) ,
alpha(alpha) ,
max_iter(max_iter) ,
tol(tol) ,
normalize(normalize) {;}



bool Lasso::valid(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y){
    if (X.rows() != y.rows()) {
        return false;
    }
    return true;
}

bool Lasso::fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y){
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
        cost = 0;

        VectorXd y_pred = X_norm * intercept_coef_;
        VectorXd error = y - y_pred;
        
        cost = error.dot(error)/X_norm.rows();

        // VectorXd intercept_coef_2 = intercept_coef_.cwiseAbs().array() + EPSILON;   
        VectorXd intercept_coef_2 = intercept_coef_;   
        for(int i=0; i<intercept_coef_2.rows(); ++i) 
            if (intercept_coef_2(i) >= 0) intercept_coef_2(i) = 1; else intercept_coef_2(i) = -1;

        // grad = -2.0/X_norm.rows() * X_norm.transpose() * error + alpha * intercept_coef_.cwiseQuotient(intercept_coef_2);
        grad = -2.0/X_norm.rows() * X_norm.transpose() * error + alpha * intercept_coef_2;
        
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

VectorXd Lasso::predict(const Matrix<double, Dynamic, Dynamic> &X){
    if (X.cols() != coef_.rows()) {
        error_flag_ = 1;
        return VectorXd::Zero(1);
    }
    VectorXd pre = (X * coef_).array() + intercept_;
    return pre;
}

