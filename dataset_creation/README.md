# Dataset creation 

## caffe record creation
* Steps for caffe record creation 
```
sudo apt-get update
sudo apt-get install -y libopencv-dev
dpkg -L libopencv-dev
protoc --cpp_out="./" ./caffe_protos.proto 

sudo  g++ caffee_lmdb_create_parse.cpp -lprotobuf caffe_protos.pb.cc -lpthread -llmdb -ljsoncpp `pkg-config --cflags --libs opencv4`
 ./a.out  <PATH TO EMPTY LMDB FOLDER> <IMAGE FOLDER OF JPEG IMAGES> <COCO DATASET>.json
```

## caffe2 record creation
* Steps for caffe record creation 
```
sudo apt-get update
sudo apt-get install -y libopencv-dev
dpkg -L libopencv-dev
protoc --cpp_out="./" ./caffe2_protos.proto 

sudo  g++ caffee2_lmdb_create_parse_detection.cpp -lprotobuf caffe2_protos.pb.cc -lpthread -llmdb -ljsoncpp `pkg-config --cflags --libs opencv4`
 ./a.out  <PATH TO EMPTY LMDB FOLDER> <IMAGE FOLDER OF JPEG IMAGES> <COCO DATASET>.json
```
## TF_record creation
* For TF_Classification
```
python tf_record_creation.py
```
* For TF_Detection
```
change 'image/class/label' to 'image/object/class/label'
python tf_record_creation.py
```




# Steps to create MXNet RecordIO files using MXNet's im2rec.py script

## MXNet Installation

pip install mxnet

## Step1 : to create .lst file

python mxnet_record_creation.py --list test Dataset_path --recursive

test - name of your .lst file

Dataset_path - path to the list of image folders

--recursive - If set recursively walk through subdirs and assign an unique label to images in each folder. Otherwise only include images in the root folder and give them label 0

## Step2 : to create RecordIO files

python mxnet_record_creation.py lst_file Dataset_path

lst_file - *.lst file created using Step1

Dataset_path - path to the list of image folders
