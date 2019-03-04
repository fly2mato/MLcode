#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <list>
#include <algorithm>
#include <string>
#include <map>
#include <unordered_map>


using namespace std;


class TreeNode{
public:
    uint decision_feature;
    string ans;
    // vector<string> feature_values;
    vector<TreeNode*> child;

    TreeNode() {
        decision_feature=0;
        ans = "";
        // feature_values.clear();
        child.clear();
    }
};

class DecisionTree{
private:    
    vector<int> feature_dim;
    vector<map<string, uint>> feature_value;
    
    uint label_dim;
    map<string,uint> label_value;
    TreeNode *dt;

public:
    DecisionTree();
    ~DecisionTree(){};
    void fit(const vector<vector<string>> & X, const vector<string> & y);

    void buildTree(const vector<vector<string>>&X, const vector<string>&y, vector<uint> sample_index, vector<uint> feature_index, uint depth, TreeNode* root);

    string predict(const vector<string> &x);
};











#endif