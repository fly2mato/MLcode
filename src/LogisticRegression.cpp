#include "LogisticRegression.h"

LogisticRegression::LogisticRegression() :
    min_error(1e-10),
    max_iter(10000),
    learning_rate(0.1),
    error_flag_(-1) {}



bool fit(const Matrix<double, Dynamic, Dynamic> &X, const VectorXd &y);
VectorXd predict(const Matrix<double, Dynamic, Dynamic> &X);