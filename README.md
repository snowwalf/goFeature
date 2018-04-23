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

## Performance
### Benchmark
> Test on NVIDIA P4, 512

|feature number|parallel|GPU load|average delay|QPS|
|:---|:---|:---|:---|:---|
|10000|1|1%|13.57ms|73.74|
|10000|5|1%|66.83ms|74.76|
|10000|10|1%|131.41ms|76.03|
|100000|1|6%|31.15ms|32.10|
|100000|5|9%|95.81ms|52.11|
|100000|10|9%|192.74ms|51.68|
|500000|1|26%|36.83ms|27.15|
|500000|5|46%|89.91|55.49|
|500000|10|47%|179.71|55.43|
|1000000|1|36%|48.67ms|20.55|
|1000000|5|55%|150.26ms|33.21|
|1000000|10|55%|293.49ms|33.55|