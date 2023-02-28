# Dataset creation 

## caffe record creation
* Steps for caffe record creation 
```
sudo apt-get update
sudo apt-get install -y libopencv-dev
dpkg -L libopencv-dev
protoc --cpp_out="./" ./caffe_protos.proto 

sudo  g++ caffee_lmdb_create_parse.cpp -lprotobuf caffee_protos.pb.cc -lpthread -llmdb -ljsoncpp `pkg-config --cflags --libs opencv4`
```

## caffe2 record creation
* Steps for caffe record creation 
```
sudo apt-get update
sudo apt-get install -y libopencv-dev
dpkg -L libopencv-dev
protoc --cpp_out="./" ./caffe2_protos.proto 

sudo  g++ caffee2_lmdb_create_parse.cpp -lprotobuf caffee_protos.pb.cc -lpthread -llmdb -ljsoncpp `pkg-config --cflags --libs opencv4`
```
