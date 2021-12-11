CXX := /usr/local/cuda-10.2/bin/nvcc
TARGET := conv
CUDNN_PATH := /usr/local/cuda-10.2
HEADERS := -I $(CUDNN_PATH)/include 
LIBS := -L $(CUDNN_PATH)/lib64 
CXXFLAGS := -arch=sm_52  -gencode=arch=compute_52,code=sm_52  -gencode=arch=compute_60,code=sm_60  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70  -gencode=arch=compute_75,code=sm_75 -Xptxas "-v -dlcm=ca" -Xcompiler="-fPIC"
all: conv

conv: *.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) *.cu -o $(TARGET) -lcudnn
.phony: clean

clean:
	rm $(TARGET) || echo -n ""
