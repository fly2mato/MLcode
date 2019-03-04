#include "DecisionTree.h"

DecisionTree::DecisionTree(){
    feature_dim.clear();
    feature_value.clear();
    label_dim = 0;
    label_value.clear();
    dt = nullptr;
}


void DecisionTree::fit(const vector<vector<string>> &X, const vector<string> &y){
    //todo encode过程
    
    //使用信息增益比



    for(uint i=0; i<y.size(); ++i){
        if (label_value.find(y[i])==label_value.end()) {
            label_value[y[i]] = label_dim;
            label_dim++;
        }
    }   

    //每一个特征的取值维度
    map<string, uint> feature_map;
    feature_dim.resize(X[0].size());
    for(uint j=0; j<X[0].size(); ++j){
        feature_map.clear();
        for(uint i=0; i<X.size(); ++i){
            if (feature_map.find(X[i][j])==feature_map.end()){
                feature_map[X[i][j]] = feature_dim[j];
                feature_dim[j]++;
            }
        }
        feature_value.push_back(feature_map);
    }

    dt = new TreeNode;

    vector<uint> sample_index;
    vector<uint> feature_index;
    for(uint i=0; i<y.size(); ++i) sample_index.push_back(i);
    for(uint j=0; j<X[0].size(); ++j) feature_index.push_back(j);

    buildTree(X, y, sample_index, feature_index, 0, dt);

}


void DecisionTree::buildTree(const vector<vector<string>>&X, const vector<string>&y, vector<uint> sample_index, vector<uint> feature_index, uint depth, TreeNode * root){
    
    //计算H(D)
    double HD = 0;
    unordered_map<string, uint> count;
    for(uint i=0; i<sample_index.size(); ++i) count[y[sample_index[i]]]++;
    for(auto i : label_value){
        double p = count[i.first]*1.0/sample_index.size();
        if (p<1e-6) continue;
        HD -= p*log2(p);
    }

    uint max_count = 0;
    string max_y = "";
    for(auto c:count){
        if (c.second>max_count) {
            max_count = c.second;
            max_y = c.first;
        }
    }

    // cout << '#' << depth << ':' << max_y << ',' << max_count << endl;
    // cout << HD << ',' << feature_index.size() << endl;

    if (feature_index.size()==0 || HD<1e-6){
        root->ans = max_y;
        return ;
    }


    double gDA = 0.0;
    double max_gDA = 0.0;
    uint max_feature = feature_index[0];
    for(auto j:feature_index){
        //每个特征取值情况下，分别对label取值计数
        vector<vector<uint>> c(feature_dim[j], vector<uint>(label_dim, 0));  
        vector<uint> sumc(feature_dim[j], 0); //每个特征的取值情况，共有多少个样本       
        for(auto i:sample_index){
            int feature_x = feature_value[j][X[i][j]]; //第j个特征，在样本点上的取值，编码
            int label_y = label_value[y[i]]; //样本点的标签，编码
            c[feature_x][label_y]++;
            sumc[feature_x]++;
        }

        double HDa = 0.0; //H(D,a)
        double HAD = 0.0; //用于计算信息增益比 的 分母
        uint sum_i = 0;

        for(auto a:c){
            double pa = sumc[sum_i]*1.0/sample_index.size();
            double HD_a = 0.0; //H(D|a)
            for(auto ay:a){
                if (ay==0) continue;
                HD_a -= ay*1.0/sumc[sum_i] * log2(ay*1.0/sumc[sum_i]);
            }
            HDa += pa * HD_a;
            sum_i++; 
            if (pa<1e-6) continue;
            HAD -= pa*log2(pa);  
        }
        gDA = (HD-HDa)/HAD;
        if (gDA > max_gDA) {
            max_gDA = gDA;
            max_feature = j;
        } 

    }
    root->decision_feature = max_feature;
    vector<uint> next_feature_index;
    for(auto j: feature_index){
        if (j!= max_feature) next_feature_index.push_back(j);
    }
    root->child.resize(feature_value[max_feature].size());
    for(auto a: feature_value[max_feature]){
        TreeNode * pt = new TreeNode;
        // root->feature_values.push_back(a.first);
        root->child[a.second] = pt;

        vector<uint> next_sample_index;
        for(auto i : sample_index){
            if (X[i][max_feature] == a.first) next_sample_index.push_back(i);
        }

        if (next_sample_index.size() == 0) {
            pt->ans = max_y;
            continue;
        }

        buildTree(X, y, next_sample_index, next_feature_index, depth+1, pt);
    }

}

string DecisionTree::predict(const vector<string>& x){
    TreeNode *pt = dt;
    while(pt->ans==""){
        pt = pt->child[feature_value[pt->decision_feature][x[pt->decision_feature]]];
    }
    return pt->ans;
}