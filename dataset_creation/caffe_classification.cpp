#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <lmdb.h>
#include "caffe_protos.pb.h"
#include <experimental/filesystem>
using namespace std;
using namespace cv;

#include <dirent.h>
#define CACHE_SIZE 20UL * 1024UL * 1024UL * 1024UL

// Create LMDB record
int main() {
    // Define paths
    string image_folder_path = "/media/fiona/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/";
    string lmdb_path = "/media/fiona/record_creation/caffe_classification_labels/";
    string label_file_path = "/media/fiona/labels_list.txt";

    // Load labels
    // ifstream label_file(label_file_path.c_str());
    vector<string> labels;
    string line;

    MDB_env* env;
    MDB_dbi dbi;
    MDB_val key, data;
    MDB_txn *txn;
    mdb_env_create(&env);
    mdb_env_set_mapsize(env, CACHE_SIZE);
    mdb_env_open(env, lmdb_path.c_str(), 0, 0664);
    
    mdb_txn_begin(env, NULL, 0, &txn);
    mdb_dbi_open(txn, NULL, 0, &dbi);
    int count = 0;

    char path[1000];
    struct dirent *dp;
    DIR *dir = opendir(image_folder_path.c_str());
    char *fileName[1000];
    // Unable to open directory stream
    if (!dir)
        return 0;
    
    char substr[] = ".jpg";

    while ((dp = readdir(dir)) != NULL)
    {
        if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
        {
            // cout << dp->d_name;
            string image_name = string(dp->d_name);
            string image_path = image_folder_path + image_name;
            std::cerr<<"\n image_path  "<< image_path;
            Mat image = imread(image_path);

            FILE * fp = fopen (image_path.c_str() , "rb");
            fseek(fp,0L,SEEK_END);
            int max_size=ftell(fp);
            fseek(fp,0L,SEEK_SET);
            char *r_image=(char*)malloc(max_size*sizeof(char));
            fread(r_image,1,max_size,fp);  

            if (image.empty()) {
                cerr << "Failed to load image: " << image_path << endl;
                continue;
            }
            string image_label;
            ifstream label_file(label_file_path.c_str());
            if (label_file.is_open())
            { 
                while (getline(label_file, line)) {
                    if(-1 != line.find(image_name))
                    {
                        for (auto x : line) 
                        { 
                            if (x == ' ') 
                                image_label = ""; 
                            else
                                image_label = image_label + x; 
                        }
                    }
                }
                label_file.close();
            }      
            else 
                cout << "Unable to open LABELS FILE"; 
            string label = image_label;
            image_label.clear();
            cout << "Label " << label << "\n";
            count++;
            caffe_protos::Datum datum;
            datum.set_label(stoi(label));
            datum.set_channels(image.channels());
            datum.set_height(image.rows);
            datum.set_width(image.cols);
            datum.set_data(r_image, max_size * sizeof(char));

            string value;
            datum.SerializeToString(&value);

            // string i = image_name;
            // key.mv_size = i.length();
            // key.mv_data = (void *)i.c_str();
            // data.mv_size = value.size();

            string i = image_name;
            key.mv_size = i.length();
            key.mv_data = (void *)i.c_str();
            data.mv_size = value.length();
            data.mv_data = (char *)value.c_str();
            cout << "MV KEYSIZE: " << key.mv_size << endl;
            cout << "MV KEYDATA: " << string((char *) key.mv_data) << endl;    
            cout << "DATA SIZE.....: " << data.mv_size << endl;
            mdb_put(txn, dbi, &key, &data, 0);
        }
    }
    mdb_txn_commit(txn);
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
    return 0;
}
