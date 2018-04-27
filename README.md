# goFeature
 
**Master** [![Build Status](https://travis-ci.org/snowwalf/goFeature.svg?branch=master)](https://travis-ci.org/snowwalf/goFeature)
**Dev**[![Build Status](https://travis-ci.org/snowwalf/goFeature.svg?branch=dev)](https://travis-ci.org/snowwalf/goFeature)
 
 golang library for feature search on cublas
 **Limitation**
 * Only support noramlized feature
 * Only use cosine distance to compare vectors

## Dependency

```
go get github.com/unixpickle/cuda
go get github.com/unixpickle/cuda/cublas
```

## Documents
* [unixpickle/cuda](https://godoc.org/github.com/unixpickle/cuda)
* [unixpickle/cublas](https://godoc.org/github.com/unixpickle/cuda/cublas)


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
> Test on NVIDIA P4, dimension = 512, precision = 4, feature_num = 800W, block_size =32MB, cores = 100 (GPU Mem:Host Mem = 1:5), search_parallel = 1

|batch|average delay|QPS|
|:---|:---|:---|
|1|2.271s|0.440|
|2|2.299s|0.869|
|3|2.375s|1.263|
|4|2.481s|1.612|
|5|2.412s|2.073|
|6|2.446s|2.452|
|7|2.53s|2.757|
|8|2.678s|2.987|
|9|2.620s|3.434|
|10|2.799s|3.573|

**Tips**
* average GPU load: 65%
* GPU Mem： 3.5G
* Host Mem: 17G

### Search Parallel
> Test on NVIDIA P4, dimension = 512, precision = 4, feature_num = 100W, block_size =16MB, cores = 125 (GPU Mem:Host Mem = 1:1), batch = 1

|parallel|average delay|QPS|
|:---|:---|:---|
|1|66.8ms|149.6|
|2|88.3ms|226.88|
|3|130.4ms|229.32|
|4|173.2ms|229.16
|5|218.2ms|227.9|
|6|265.1ms|227.9|
|7|303.8ms|229.3|
|8|346.2ms|230.5|
|9|391.7ms|228.8|
|10|436.4ms|228.8|

**Tips**
* average GPU load: 50%
* GPU Mem： 2.3G
* Host Mem: 3.1G