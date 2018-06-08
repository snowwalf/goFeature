FROM nvidia/cuda:8.0-cudnn6-devel

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
# 这两个 NVIDIA source list 更新存在问题
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates telnet wget curl vim unzip git && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV GOLANG_VERSION 1.8.7
ENV GOLANG_DOWNLOAD_URL https://dl.google.com/go/go1.8.7.linux-amd64.tar.gz
ENV GOLANG_DOWNLOAD_SHA256 de32e8db3dc030e1448a6ca52d87a1e04ad31c6b212007616cfcc87beb0e4d60

RUN curl -fsSL "$GOLANG_DOWNLOAD_URL" -o golang.tar.gz \
        && echo "$GOLANG_DOWNLOAD_SHA256  golang.tar.gz" | sha256sum -c - \
        && tar -C /usr/local -xzf golang.tar.gz \
        && rm golang.tar.gz

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"
ENV WORKSPACE /workspace/goFeature
RUN go get -d github.com/unixpickle/cuda && go get -d github.com/unixpickle/cuda/cublas
RUN go get -d gorgonia.org/cu && wget https://raw.githubusercontent.com/gorgonia/cu/v0.9.0-working/blas/consopt.go -O /go/src/gorgonia.org/cu/blas/consopt.go && \
    wget http://oqgascup5.bkt.clouddn.com/blas.go -O /go/src/gorgonia.org/cu/blas/blas.go
RUN go get -d github.com/gonum/blas
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV CUDA_PATH "/usr/local/cuda-8.0"
ENV CPATH "$CUDA_PATH/include/"
ENV CGO_LDFLAGS "$CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcurand.so"
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_PATH/lib64:$CUDA_PATH/lib64/stubs
ENV PATH $PATH:/workspace/goFeature/bin