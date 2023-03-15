#include <stdio.h>
#include "lmdb.h"
#include <string.h>  
#include <iostream>
#include "caffe_protos.pb.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <jsoncpp/json/value.h>
#include <jsoncpp/json/json.h>

#include <dirent.h>

#include <unistd.h>
#define GetCurrentDir getcwd

#define CACHE_SIZE 1UL * 1024UL * 1024UL * 1024UL

#define E(expr) CHECK((rc = (expr)) == MDB_SUCCESS, #expr)
#define CHECK(test, msg); ((test) ? (void)0 : ((void)fprintf(stderr, \
    "%s:%d: %s: %s\n", __FILE__, __LINE__, msg, mdb_strerror(rc)), abort()))
    
#define ANNOTATION_TYPE_BBOX 0    

using namespace std;
using namespace cv;


// This function parses all the data values retrieved from the LMDB records
void parse_Image_Metadata_Protos(caffe_protos::AnnotatedDatum &annotatedDatum_protos, 
                     char *imgOutput, int outImgCnt)
{
    // Checking image Datum
    int chk_image_datum = annotatedDatum_protos.has_datum(); 
    //cout << "Check Image Datum Metadata: " << chk_image_datum << endl;
    
    if(chk_image_datum)
    {
        caffe_protos::Datum image_datum = annotatedDatum_protos.datum();
        
        // Parsing width of image
        int dim_width = image_datum.width();
        cout << "Width: " << dim_width << endl;

        // Parsing height of image
        int dim_height = image_datum.height();
        cout << "Height: " << dim_height << endl;
            
        // Parsing channels of image
        int dim_channels = image_datum.channels();
        cout << "Channels: " << dim_channels << endl; 
        
        bool chk_image_datum_data = image_datum.has_data();
        //cout << "Check Image Datum Byte: " << chk_image_datum << endl;
        
        if(chk_image_datum)
        {
            // Parsing Image bytes
            string image_bytes = image_datum.data();
            cout << "Image bytes: " << image_bytes.length() << endl;
                    
            // Buffer to hold image bytes as uchar vector array
            vector<uchar> vec_data(image_bytes.c_str(), image_bytes.c_str() + image_bytes.size());
                        
            // Decoding Image vector<uchar> bytes
            Mat cv_img = imdecode(vec_data, -1);
            if (!cv_img.data)
                cout << "Could not decode Image bytes" << endl;
            else
            {            
                // Writing images to Folder
                char output_path[100] = {};   
                char text[80];   
                strcpy(output_path, imgOutput);
                sprintf(text,"/IMAGE_OUT_%d", outImgCnt);
                strcat(output_path, text);
                strcat(output_path, ".jpg");
                imwrite(output_path, cv_img);
            }
        }
        else
        {
            cout << "Could not parse Image Datum Bytes" << endl;
        }
    }
    else
    {
        cout << "Image Datum Metadata Parsing Failed" << endl;
    }
    
    caffe_protos::AnnotationGroup annotGrp_protos = annotatedDatum_protos.annotation_group(0);
    
    int boundBox_size = annotGrp_protos.annotation_size();;
    //cout << "Annotation Size: " << boundBox_size << endl;
    
    if(boundBox_size != 0)
    {
        int boundIter = 0;
        for(int i = 0; i < boundBox_size; i++)
        {
            caffe_protos::Annotation annot_protos = annotGrp_protos.annotation(i);
            caffe_protos::NormalizedBBox bbox_protos = annot_protos.bbox();
            
            // Parsing the bounding box points using Iterator
            int boundBox_xMin = bbox_protos.xmin();
            cout << "Bounding Box XMIN: " << boundBox_xMin << endl;
                
            int boundBox_yMin = bbox_protos.ymin();
            cout << "Bounding Box YMIN: " << boundBox_yMin << endl;
                
            int boundBox_width = bbox_protos.xmax();
            cout << "Bounding Box Width: " << boundBox_width << endl;

            int boundBox_height = bbox_protos.ymax();
            cout << "Bounding Box Height: " << boundBox_height << endl;
            
            int boundBox_label = bbox_protos.label();
            cout << "Bounding Box label: " << boundBox_label << endl;
                
            cout << "*********Bounding Box Point(Count:-" << i << ")*************" << endl;
        }
    }
    else
    {
       cout << "Bounding Box Size is Zero. No values Present" << endl;
    }
}

// This function reads all the LMDB records from a <file>.mdb file to retrieve all the key-value pairs 
void read_caffe_lmdb_records(char *dbFolder, int file_bytes, char *imgOutput)
{
	int rc;
	MDB_env *env;
	MDB_dbi dbi;
	MDB_val key, data;
	MDB_txn *txn;
	MDB_cursor *cursor;

    // Creating an LMDB environment handle
	E(mdb_env_create(&env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database. 
    E(mdb_env_set_mapsize(env, file_bytes));
    // Opening an environment handle.
	E(mdb_env_open(env, dbFolder, 0, 0664));
    // Creating a transaction for use with the environment.
	E(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	// Opening a database in the environment. 
	E(mdb_dbi_open(txn, NULL, 0, &dbi));
    
    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
	E(mdb_cursor_open(txn, dbi, &cursor));
    
    int outImgCnt = 0;
    
    // Retrieve by cursor. It retrieves key/data pairs from the database
	while((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0)
    {		
        // Reading the key value for each record from LMDB                                                                                                                                                                                                       
        cout << "KEY SIZE: "  << key.mv_size << endl;                                                                           
        cout << "KEY DATA: " << (char *)key.mv_data << endl;                                                                                                                                                                                                        
           
        // Reading the data value for each record from LMDB       
        cout << "DATA SIZE: " << data.mv_size << endl;                                                                            

        // Parsing the Image, Label and Bounding Box Protos using the key and data values.
        // Read from LMDB records
        caffe_protos::AnnotatedDatum annotatedDatum_protos;
        annotatedDatum_protos.ParseFromArray((char *)data.mv_data, data.mv_size);
        parse_Image_Metadata_Protos(annotatedDatum_protos, imgOutput, outImgCnt);
        outImgCnt++;
        
        cout << "########################################### \n\n" << endl;
	}
    
    // Closing all the LMDB environment and cursor handles
	mdb_cursor_close(cursor);
	mdb_txn_abort(txn);
	mdb_close(env, dbi);
	mdb_env_close(env);
}

// This function parses the COCO dataset which is in JSON format. It parses image annotations
void parse_Coco_dataset(caffe_protos::AnnotationGroup* annoted_group_protos, 
                  char *cocoDataset, char *imageId)
{
    cout << "Parsing COCO Dataset" << endl;
    string annotation_file = cocoDataset;
    std::cerr<<annotation_file<<"\n\n";
    // Reading the JSON (COCO)file
	ifstream fin;
	fin.open(annotation_file, std::ios::in);

	string str;
	str.assign(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    
    Json::Reader reader;
    Json::Value root;

    if (reader.parse(str, root) == false)
        cout << "Failed to parse Json: " << reader.getFormattedErrorMessages() << endl;
    
    // Parsing the JSON file with the id as annotations
    Json::Value annotation = root["annotations"];

    int id = atoi(imageId);   
    auto var = 0;
    
    // Iterating the JSON file with the id as "image_id" to get all the bounding box values
    // and label values for a given input image
    for (auto iterator = annotation.begin(); iterator != annotation.end(); iterator++) 
    {
        int img_id = (*iterator)["image_id"].asInt();
        
        if(img_id == id)
        {
            // Adding the bounding box Protos values into Annotation Protos structure.
            caffe_protos::Annotation* annotate_image_protos = annoted_group_protos->add_annotation();
            caffe_protos::NormalizedBBox* bbox_image_protos = annotate_image_protos->mutable_bbox();
            
            var = (*iterator)["bbox"][0].asFloat();           
            cout << "xMIN: " << var << endl;
            bbox_image_protos->set_xmin(var);
            
            var = (*iterator)["bbox"][1].asFloat();
            cout << "yMIN: " << var << endl;
            bbox_image_protos->set_ymin(var);
            
            var = (*iterator)["bbox"][2].asFloat();
            cout << "width: " << var << endl;
            bbox_image_protos->set_xmax(var);
            
            var = (*iterator)["bbox"][3].asFloat();
            cout << "height: " << var << endl;
            bbox_image_protos->set_ymax(var);
            
            var = (*iterator)["category_id"].asInt();
            cout << "label: " << var << endl;                             
            bbox_image_protos->set_label(var);
            
            cout << "------------------------------------" << endl;
        }
    }
}

// This function creates LMDB record of images where records has the following attributes:
// a) Image Metadata
// b) Bounding Box values and
// c) Image labels
int write_caffe_lmdb_records(char * dbFolder, char *imgFile, char *cocoDataset) 
{
    int rc;
    string image_path ="";
    string image_byte_data = "";
    string temp_path ="/media/swetha/unittest_script/new_dataset/caffe1/";
    
    // Environment handles for LMDB record creation, writing and reading
    MDB_env *env;
    MDB_dbi dbi;
    MDB_txn *txn;
    MDB_val key, data, rdata;
    MDB_cursor *cursor;
    
    // Finds the current working directory
    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    string current_working_dir(buff);
    //cout << "CURRENT DIRECTORY: " << current_working_dir << endl;        
    
    // Creating structure for caffe PROTOS for Image, Label and Bounding Box
    caffe_protos::AnnotatedDatum annotatedDatum_protos = caffe_protos::AnnotatedDatum();
    caffe_protos::Datum* image_datum_protos = annotatedDatum_protos.mutable_datum();
    caffe_protos::AnnotationGroup* annoted_group_protos = annotatedDatum_protos.add_annotation_group();
    
    // Creating an LMDB environment handle
    E(mdb_env_create(&env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database.
    E(mdb_env_set_mapsize(env, CACHE_SIZE));
    // Opening an environment handle.
    E(mdb_env_open(env, dbFolder, 0, 0664));
    // Creating a transaction for use with the environment.
    E(mdb_txn_begin(env, NULL, 0, &txn));
    // Opening a database in the environment.
    E(mdb_dbi_open(txn, NULL, 0, &dbi));

    // Opening a directory to read all images from a given input folder
    struct dirent *dp;
    DIR *dir = opendir(imgFile);
    if (!dir)
    {
        cout << "Opening and Reading Image from Input Folder Failed" << endl;
        return -1;
    }

    // Reading the images from a folder, creating PROTOS structure of images, labels and
    // bounding box and writing these PROTOS structure as LMDB records for all the images
    while ((dp = readdir(dir)) != NULL)
    {
        if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
        {
            cout << "DIRECTORY FILE OPEN: " << dp->d_name << endl;  
            
            // Parsing COCO dataset for a given image_id 
            parse_Coco_dataset(annoted_group_protos, cocoDataset, dp->d_name);

            // image_path =  current_working_dir + "/" + imgFile + dp->d_name;
            string temp = dp->d_name;
            image_path =   imgFile + temp;

            // /media/swetha/unittest_script/new_dataset/caffe1/
            cout << "IMAGE PATH: " << image_path << endl<<imgFile;
                
            // Reading image from a file an creating a MAT structure    
            Mat img = imread(image_path);   
            FILE * fp = fopen (image_path.c_str() , "rb");
            fseek(fp,0L,SEEK_END);
            int max_size=ftell(fp);
            fseek(fp,0L,SEEK_SET);
            char *r_image=(char*)malloc(max_size*sizeof(char));
            fread(r_image,1,max_size,fp);
            


            if (!img.data)
            {
                cout << "Reading Image from specfied File Path Failed" << endl;
                return -1;
            }      
               
            // Adding image metadata in an Image Datum PROTOS structure   
            image_datum_protos->set_width(img.size().width);
            image_datum_protos->set_height(img.size().height);
            image_datum_protos->set_channels(img.channels());
            
            // Encoding the image bytes into Image Datum PROTOS structure
            vector<uchar> buf;
            imencode(".jpg",img,buf);
            image_datum_protos->set_data(r_image, max_size * sizeof(char));
                
            // Creating a serailized object of images, label and Bounding box PROTOS structure    
            annotatedDatum_protos.SerializeToString(&image_byte_data);
                
            //cout << "CHECK IMAGE BYTES: " << image_datum_protos->has_data() << endl;
            //cout << "NOTICE IMAGE DATAS: " << image_datum->data() << endl;
                
            // Creating key value for LMDB record    
            string strKey = dp->d_name; 
            key.mv_size = strKey.length();
            key.mv_data = (void *)strKey.c_str();

            // Creating data value for LMDB record
            data.mv_size = image_byte_data.length();
            data.mv_data = (char *)image_byte_data.c_str();
                
            cout << "MV KEYSIZE: " << key.mv_size << endl;
            cout << "MV KEYDATA: " << string((char *) key.mv_data) << endl;    
            cout << "DATA SIZE.....: " << data.mv_size << endl;
                        
            // Putting the LMDB record of keys and image values into the LMDB database           
            E(mdb_put(txn, dbi, &key, &data, 0));  
            
            // Clearing annotated image protos from annotated group protos
            annoted_group_protos->clear_annotation();
            
            cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n" << endl;
       }
    }
    E(mdb_txn_commit(txn)); 

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);

    return 0;
}

int main(int argc,char * argv[])
{
    char *dbFolder = argv[1];
    char *image_Folder = argv[2];  
    char *cocoDataset = argv[3];
    
    /*
     * Here image_Folder is the folder which contains multiple jpg images as is used as an input.
     */
    write_caffe_lmdb_records(dbFolder, image_Folder, cocoDataset);
    
    string tmp1 = string(dbFolder) + "/data.mdb";   
    string tmp2 = string(dbFolder) + "/lock.mdb";
    int file_size, file_size1;    
    for(int i = 0; i < 2; i++)
    {      
        if(i == 0){
            ifstream in_file(tmp1, ios::binary);
            in_file.seekg(0, ios::end);
            file_size = in_file.tellg();
            //cout<<"Size of the file is"<<" "<< file_size<<" "<<"bytes" << endl;
        }        
        else{           
            ifstream in_file1(tmp2, ios::binary);
            in_file1.seekg(0, ios::end);
            file_size1 = in_file1.tellg();
            //cout<<"Size of the file is"<<" "<< file_size1<<" "<<"bytes" << endl;
        }   
    }   
    int file_bytes = file_size + file_size1;
    
    /*
     * Here image_Folder is an empty folder where the result images are written 
     */
    //read_caffe_lmdb_records(dbFolder, file_bytes, image_Folder);
	return 0;
}
