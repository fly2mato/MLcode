#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <list>
#include <algorithm>
#include <string>
#include <time.h>
#include <stdlib.h>

#include "DecisionTree.h"

using namespace std;


int main(){
    srand((unsigned int)(time(NULL)));
    fstream fid;
    fid.open("../mytest/DTdata.txt");
    vector<string> y;
    vector<vector<string>> X;
    string str;
    vector<string> sstr;
    for(int i=0; i<14; ++i){
        fid >> str;
        y.push_back(str);
        sstr.clear();
        for(int j=0; j<4; ++j){
            fid >> str;
            sstr.push_back(str);
        }
        X.push_back(sstr);
    }

    DecisionTree DTs;
    DTs.fit(X,y); 
    for(auto x:X){
        cout << DTs.predict(x) << endl;
    }
    return 1;
}

