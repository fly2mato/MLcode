#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <list>
#include <algorithm>
#include <string>
#include <time.h>
#include <stdlib.h>

#include "LinearRegression.h"
#include "LogisticRegression.h"

using namespace std;


void test_ridge();
void test_lasso();
void test_lr();

int main(){
    srand((unsigned int)(time(NULL)));

    // vector<vector<double>> X;
    // vector<double> Y;
    // vector<double> x;
    // X.clear();
    // Y.clear();

    // Matrix<double, Dynamic, Dynamic> X_eigen = MatrixXd::Zero(100000, 50);    
    // VectorXd Y_eigen = VectorXd::Zero(100000);

    // vector<double> real_coef;
    // for(int j=0; j<50; ++j) real_coef.push_back((rand() % (10+10+1))-10); //[-10,10]

    // double y;
    // for(int i=0; i<100000; ++i) {
    //     x.clear();
    //     y = 1;
        
    //     VectorXd x_eigen;
    //     for(int j=0; j<50; ++j){
    //         x.push_back(100*rand()/double(RAND_MAX));            
    //         y += x.back() * real_coef[j];
    //         X_eigen(i,j) = x.back();
    //     }
    //     X.push_back(x);
    //     Y.push_back(y);
    //     Y_eigen(i) = y;
    // } 

    // //LinearRegression lr1;
    // Ridge lr1(0, 0.01, 1000, 1e-3, false);
    // auto t10 = clock();
    // lr1.fit(X_eigen, Y_eigen);
    // auto t11 = clock();
    // // cout << lr1.intercept_coef_(0) << " 1" << endl;
    // // for(int i=0; i<20; ++i){
    // //     cout << lr1.intercept_coef_(i+1) << ',' << real_coef[i] << endl;
    // // }

    // // Matrix<double, 4, 3> A;
    // // A << 1,2,3,
    // //      4,5,6,
    // //      7,8,9,
    // //      2,4,6;
    
    // // Vector3d B;
    // // B << 1,2,3;

    // // Matrix<double, 4, 3> C = ((A.array().rowwise() - B.transpose().array())).rowwise() / B.transpose().array();
    // // cout << C * B << endl;// / B.transpose().array() << endl;
    // // cout << (((A.array().rowwise() - B.transpose().array())).rowwise() / B.transpose().array()) * B << endl;
    // //cout << A * B<< endl;
    
    // LinearRegression lr;
    // auto t0 = clock();
    // lr.fit(X,Y);
    // auto t1 = clock();
    // cout << lr.intercept_ << "," << lr1.get_intercept() << endl;
    // for(int i=0; i<20; ++i){
    //     // cout << lr.coef_[i] << ',' << real_coef[i] << endl;
    //     cout << lr.coef_[i] << ',' << lr1.get_coef()(i) << endl;
    // }
    // // cout << lr.intercept_ << "," << lr1.intercept_coef_(0) << endl;
    // // for(int i=0; i<20; ++i){
    // //     // cout << lr.coef_[i] << ',' << real_coef[i] << endl;
    // //     cout << lr.coef_[i] << ',' << lr1.intercept_coef_(i+1) << endl;
    // // }
    // cout << "vector running time(sec): " << (double)(t1-t0)/CLOCKS_PER_SEC << endl;
    // cout << "Eigen running time(sec): " << (double)(t11-t10)/CLOCKS_PER_SEC << endl;

    // test_ridge();
    //test_lasso();
    test_lr();



    return 1;
}

void test_lr(){
    LogisticRegression lr(0.001, 100000, 1e-3, false);
    ifstream fid;
    fid.open("../mytest/lrdata.txt");
    int m,n;
    fid >> m >> n;
    m = 100;
    Matrix<double, Dynamic, Dynamic> X=Matrix<double,Dynamic,Dynamic>::Zero(m,n);
    VectorXd y=VectorXd::Zero(m);
    lr.read_data(fid, m,n, X,y);
    fid.close();

    lr.fit(X,y);
    cout << lr.get_coef() << endl << lr.get_intercept() << endl;
    cout << lr.predict(X) - y << endl;
}


void test_ridge(){

    Ridge lr(0.2, 0.1, 100000, 1e-3, true);
    Matrix<double, 3, 2> X;
    Vector3d y;
    X << 0,0, 
         0,0,
         1,1;
    y << 0,.1,1;
    lr.fit(X, y);
    cout << lr.get_coef().transpose() << endl;
    cout << lr.get_intercept() << endl;

    Matrix2d C = Matrix2d::Identity(2,2);
    cout << (X.transpose()*X + lr.alpha*C).inverse()*(X.transpose()*y) << endl;
}

void test_lasso(){

    Lasso lr(0.1, 0.1, 100000, 1e-9, true);
    Matrix<double, 3, 2> X;
    Vector3d y;
    X << 0,0, 
         1,1.1,
         2,2;
    y << 0,1,2;
    lr.fit(X, y);
    cout << lr.get_coef().transpose() << endl;
    cout << lr.get_intercept() << endl;

    //Matrix2d C = Matrix2d::Identity(2,2);
    //cout << (X.transpose()*X + lr.alpha*C).inverse()*(X.transpose()*y) << endl;
}
