# goFeature
 
**Master** [![Build Status](https://travis-ci.org/snowwalf/goFeature.svg?branch=master)](https://travis-ci.org/snowwalf/goFeature)
**Dev**[![Build Status](https://travis-ci.org/snowwalf/goFeature.svg?branch=dev)](https://travis-ci.org/snowwalf/goFeature)
 
 golang library for feature search on cublas
 **Limitation**
 * Only support noramlized feature
 * Only use cosine distance to compare vectors

## Dependency

```
go get  github.com/gonum/blas
go get  github.com/snowwalf/cu 
```


## Docker Compile Environment

```
docker build -t <image_name> -f docker/Dockerfile .
nvidia-docker run -dt -v <pwd>:/go/src/github.com/snowwalf/goFeature --name <container_name> <image_name> /bin/bash
docker exec -it <container_name> /bin/bash
```

## Build Demo
go build -tags 'cublas'

## Test
### Func Test

```
 go test -tags 'cublas' .
```
 
## Performance Benchmark
### Batch
> Test on NVIDIA P4, dimension = 512, precision = 4, block_size =32MB, parallel = 1

|feature num |batch|average delay|
|:----|:---|:---|
|100k|1|0.001547s|
|100k|2|0.001623s|
|100k|5|0.001685s|
|100k|10|0.001716s|
|500k|1|0.007649s|
|500k|2|0.007791s|
|500k|5|0.007918s|
|500k|10|0.007962s|
|1m|1|0.016649s|
|1m|2|0.016054s|
|1m|5|0.016464s|
|1m|10|0.016103s|
|2m|1|0.034573s|
|2m|2|0.036185s|
|2m|5|0.037065s|
|2m|10|0.034860s|


**Tips**
* average GPU load: 95 ~ 100%