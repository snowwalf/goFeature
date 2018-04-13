# goFeature
 golang library for feature search on cublas

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

## Benchmark
### Search Benchmark

```
 go test -tags 'cublas' -bench='BenchmarkSearch' -benchtime=3s -run=none
```