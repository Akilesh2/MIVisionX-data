#include <stdio.h>
#include "lmdb.h"
#include <string.h>
#include <fstream>
#include <iostream>
#include "caffe2_protos.pb.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <dirent.h>

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#define BUF_SIZE 1024
#define CACHE_SIZE 20UL * 1024UL * 1024UL * 1024UL

#define E(expr) CHECK((rc = (expr)) == MDB_SUCCESS, #expr)
#define CHECK(test, msg); ((test) ? (void)0 : ((void)fprintf(stderr, \
    "%s:%d: %s: %s\n", __FILE__, __LINE__, msg, mdb_strerror(rc)), abort()))

using namespace std;
using namespace cv;

int it = 0;

void store_lmdb_records(char * DB_Path, char *input_img_folder, char *image_file, char *labels_file) 
{
    int label_array[]={22,77,2,77,67,4, 10,13,32,81};
    static int jj;
    std::cerr<<"\n inside store_lmdb_records\n ";
    std::cerr<<"\n input_img_folder "<<input_img_folder;
    std::cerr<<"\n labels_file  "<<labels_file ;

    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    string current_working_dir(buff);
    //cout << "CURRENT DIRECTORY: " << current_working_dir << endl;        
    const auto& input_folder = current_working_dir;
    
    int rc;
    MDB_env *env;
    MDB_dbi dbi;
    MDB_txn *txn;
    MDB_val key, data, rdata;
    MDB_cursor *cursor;
    
    string image_path ="";
    string label_path = "";
    string image_byte_data = "";
    
    caffe2_protos::TensorProtos protos = caffe2_protos::TensorProtos();
    caffe2_protos::TensorProto *image_protos = protos.add_protos();
    caffe2_protos::TensorProto *label_protos = protos.add_protos();
    
    E(mdb_env_create(&env));

    /* Set the cache size */
    E(mdb_env_set_mapsize(env, CACHE_SIZE));
    E(mdb_env_open(env, DB_Path, 0, 0664));

    /* Put some data */
    E(mdb_txn_begin(env, NULL, 0, &txn));
    E(mdb_dbi_open(txn, NULL, 0, &dbi));
    
    std::cerr<<"\n input_folder  "<<input_folder<<"\n";
    // image_path =  input_folder + "/" + input_img_folder + "/" + image_file;
    string temp_file_name=image_file;
    image_path =  input_img_folder + temp_file_name;
    
    cout << "IMAGE PATH: " << image_path << endl;
        
    label_path = input_folder + "/" + labels_file;
    label_path =  labels_file;

    cout << "LABEL PATH: " << label_path << endl;

    string line;
    ifstream myfile (label_path);
    string image_label = "";
    if (myfile.is_open())
    {    
        while (getline(myfile,line))
        {
            cout << "LABELS FILE FOUND: " << line << '\n';
            if(-1 != line.find(image_file))
            {
                for (auto x : line) 
                { 
                   if (x == ' ') 
                       image_label = ""; 
                   else
                   {
                    std::cerr<<"\n check in file reading "<<x;
                       image_label = image_label + x; 
                   }
                }  
            }
        }
        myfile.close();
    }      
    else 
        cout << "Unable to open LABELS FILE"; 
    
    cout << "LABEL WORD: " << image_label << endl; 
    Mat img = imread(image_path);    
    FILE * fp = fopen (image_path.c_str() , "rb");
    fseek(fp,0L,SEEK_END);
    int max_size=ftell(fp);
    fseek(fp,0L,SEEK_SET);
    char *r_image=(char*)malloc(max_size*sizeof(char));
    fread(r_image,1,max_size,fp);  
    if (!img.data)
    {
        cout << "Image could not loaded" << endl;
        return;
    }    
            
    image_protos->set_data_type(caffe2_protos::TensorProto::BYTE);
    image_protos->add_dims(img.rows);
    image_protos->add_dims(img.cols);
    image_protos->add_dims(img.channels());
        
    label_protos->set_data_type(caffe2_protos::TensorProto::INT32);
    label_protos->add_int32_data(stoi(image_label));
    // label_protos->add_int32_data(image_label);

    // label_protos->add_int32_data(label_array[jj++]);;

               
    vector<uchar> buf;
    imencode(".jpg",img,buf);
    image_protos->set_byte_data(r_image, max_size * sizeof(char));

    // image_protos->set_byte_data(buf.data(), buf.size() * sizeof(uchar));       

    protos.SerializeToString(&image_byte_data);
            
    string str = image_file; 
 
    key.mv_size = str.length();
    key.mv_data = (void *)str.c_str();

    data.mv_size = image_byte_data.length();
    data.mv_data = (char *)image_byte_data.c_str();   
          
    //cout << "MV KEYSIZE: " << key.mv_size << endl;
    //cout << "IMAGE: " << string((char *) key.mv_data) << endl;
        
    //cout << "DATA SIZE.....: " << data.mv_size << endl;
    //cout << "DATA DATA.....: " << string((char *) data.mv_data) << endl; 
                
    E(mdb_put(txn, dbi, &key, &data, 0));   
    E(mdb_txn_commit(txn));    

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

void write_lmdb_records(char *DB_Path, char *input_img_path, char *input_labels_path)
{
    std::cerr<<"\n inside write_lmdb_records\n ";

    char path[1000];
    struct dirent *dp;
    DIR *dir = opendir(input_img_path);
    cout << "IMAGE PATH: " << input_img_path << endl;
    
    char *fileName[1000];

    cout << "NEAR DIRECTORY: " << endl;
    // Unable to open directory stream
    if (!dir)
        return;
    
    char substr[] = ".jpg";
    
    while ((dp = readdir(dir)) != NULL)
    {
        if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
        {
            char* found = strstr(dp->d_name, substr);           
            if(found != NULL)
            {
                cout << "ITERATION COUNT: " << it << endl;
                store_lmdb_records(DB_Path, input_img_path, dp->d_name, input_labels_path);    
                it++;
            }
 
            // Construct new path from our base path
            strcpy(path, input_img_path);
            strcat(path, "/");
            strcat(path, dp->d_name);
            
            write_lmdb_records(DB_Path, path, input_labels_path);
        }
    }

    closedir(dir);
}

void parse_Image_Protos(caffe2_protos::TensorProtos &tens_protos, char *ouput_folder_path, int img_cnt)
{
    std::cerr<<"\n inside parse_Image_Protos\n ";
    // Checking size of the protos
    int protos_size = tens_protos.protos_size();   
    if(protos_size != 0)
    { 
        caffe2_protos::TensorProto image_proto = tens_protos.protos(0);
     
        cout << "--------------IMAGE DATAS AND LABELS------------------" << endl;
        
        // Parsing width of image
        int dim_width = image_proto.dims(0);
        cout << "Width: " << dim_width << endl;

        // Parsing height of image
        int dim_height = image_proto.dims(1);
        cout << "Height: " << dim_height << endl;
        
        // Parsing channels of image
        int dim_channels = image_proto.dims(2);
        cout << "Channels: " << dim_channels << endl;

        // Parsing datatype of image
        auto datatype = image_proto.data_type(); 
        cout << "Image datatype: " << datatype << endl;
        
        // Checking if image bytes is present or not
        bool chk_byte_data = image_proto.has_byte_data();
        
        if(chk_byte_data)
        {
            string image_bytes = image_proto.byte_data();
            cout << "Image bytes: " << image_bytes.length() << endl;
            
            if(image_bytes.empty())
            {
                cout << "\nIMAGE BYTES IS NULL" << endl;
            }
            else
            {
                vector<uchar> vec_data(image_bytes.c_str(), image_bytes.c_str() + image_bytes.size());
                Mat cv_img = imdecode(vec_data, -1);
                if (!cv_img.data)
                    cout << "Could not decode Image bytes" << endl;
                
                char output_path[100] = {};   
                char text[80];    
                strcpy(output_path, ouput_folder_path);                                                                                            
                sprintf(text,"/img_OUT_%d", img_cnt);
                strcat(output_path, text);
                strcat(output_path, ".jpg");
   
                imwrite(output_path, cv_img);   
            }
        }
        else
        {
            cout << "Image Parsing Failed" << endl;
        }

        // Parsing label protos
        caffe2_protos::TensorProto label_proto = tens_protos.protos(1);
            
        // Parsing label data_type
        auto label_data_type = label_proto.data_type();
            
        // Parsing Label data size
        int label_data_size = label_proto.int32_data_size();
        if(label_data_size != 0)
        {
            // Parsing label data
            auto label_data = label_proto.int32_data(0);
            cout << "Label Data: " << label_data << endl;
        }
        
        cout << "################################ \n" << endl;
    }
    else
    {
        cout << "Parsing Protos Failed" << endl;
    }
}

void read_lmdb_records(char *DB_Path, int file_bytes_size, char *ouput_folder_path)
{
    std::cerr<<"\n inside read_lmdb_records \n ";

	int rc;
	MDB_env *env;
	MDB_dbi dbi;
	MDB_val key, data;
	MDB_txn *txn;
	MDB_cursor *cursor;

	rc = mdb_env_create(&env);
    rc = mdb_env_set_mapsize(env, file_bytes_size);
	rc = mdb_env_open(env, DB_Path, 0, 0664);
	rc = mdb_txn_begin(env, NULL, 0, &txn);
	rc = mdb_open(txn, NULL, 0, &dbi);
    
	rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
	rc = mdb_cursor_open(txn, dbi, &cursor);
    
    string str_key, str_data;
    
    int i = 1;
	while((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0)
    {
        cout << "READ ITERATIONS: " << i << endl;
        cout << "IMAGE: " << string((char *) key.mv_data) << endl;
        caffe2_protos::TensorProtos tens_protos;
        tens_protos.ParseFromArray((char *)data.mv_data, data.mv_size);
        parse_Image_Protos(tens_protos, ouput_folder_path, i);
        i++;
	}
    
	mdb_cursor_close(cursor);
	mdb_txn_abort(txn);
	mdb_close(env, dbi);
	mdb_env_close(env);
    i = 0;
}

int main(int argc,char * argv[])
{
    char *DB_Path = argv[1];
    char *input_img_path = argv[2];
    char *input_labels_path = argv[3];
      
    int dataset_type;
    int record_type = atoi(input_labels_path);
    cout << "RECORD_TYPE: " << record_type << endl; 
    
    if(record_type == 1)
        dataset_type = 1;
    else
        dataset_type = 0;
    
    if(argc < 2)
    {
        cout << "USAGE: " << endl;
        cout << "ARGUMENTS NEEDED FOR CREATING LMDB RECORDS: <EMPTY DB FOLDER> <PATH TO IMAGE FOLDER [TRAIN/VAL]> <PATH TO LABEL FILE [TRAIN/VAL]> <PASS '0' for LMDB CREATION>" << endl;
        cout << "ARGUMENTS NEEDED FOR PARSING LMDB RECORDS: <PATH TO DB FOLDER> <PATH TO EMPTY OUTPUT IMAGE FOLDER> <PASS '1' for LMDB PARSING>" << endl;
    }
    
    if(dataset_type == 0) 
        write_lmdb_records(DB_Path, input_img_path, input_labels_path);
    else
    {
        string tmp1 = string(DB_Path) + "/data.mdb";
        string tmp2 = string(DB_Path) + "/lock.mdb";
        
        int file_size, file_size1;
        for(int i = 0; i < 2; i++)
        {
            if(i == 0)
            {
                ifstream in_file(tmp1, ios::binary);
                in_file.seekg(0, ios::end);
                file_size = in_file.tellg();
            }
            else
            {
                ifstream in_file1(tmp2, ios::binary);
                in_file1.seekg(0, ios::end); 
                file_size1 = in_file1.tellg();               
            }
        }
        int file_bytes = file_size + file_size1;
        
        read_lmdb_records(DB_Path, file_bytes, input_img_path);
    }  
    
	return 0;
}
