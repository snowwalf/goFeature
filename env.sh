FS_ROOT=$(pwd)
unset GOPATH
export CUDA_PATH="/usr/local/cuda-8.0"
export CPATH="$CUDA_PATH/include/"
export CGO_LDFLAGS="$CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcurand.so"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64:$CUDA_PATH/lib64/stubs
export GOROOT=/usr/local/go
export GOPATH=$GOROOT:$FS_ROOT
export PATH=$PATH:$FS_ROOT/bin