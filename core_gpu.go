// +build cublas

package goFeature

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/gonum/blas"
	"gorgonia.org/cu"
	cublas "gorgonia.org/cu/blas"
)

const (
	defaultSearchQueueSize = 10000
)

var (
	_ctx *cu.Ctx
)

// initCuda: try to init cuda context and allocator gpu buffer
func initCuda(gpuID, gpuMemSize int) (Buffer, error) {

	var err error
	dev := cu.Device(gpuID)

	_ctx = cu.NewContext(dev, cu.SchedAuto)

	totalMem, err := dev.TotalMem()
	if err != nil {
		return nil, errors.New("fail to get the total mem of gpu " + strconv.Itoa(gpuID))
	}
	if float64(totalMem)*0.95 < float64(gpuMemSize) {
		return nil, errors.New("try to allocate too much gpu mem, only " + strconv.FormatUint(uint64(totalMem), 10) + "B*0.8 can be used")
	}

	return NewGPUBuffer(_ctx, gpuMemSize)
}

type Core struct {
	Buffer
	Handle *cublas.Standard
	Input  Buffer
	Output Buffer
	// SearchJobQueue
	Queue   chan SearchJob
	Index   int
	Version uintptr
}

func NewCore(buffer Buffer) (*Core, error) {
	var err error

	core := &Core{
		Buffer: buffer,
		Index:  -1,
		Queue:  make(chan SearchJob, defaultSearchQueueSize),
	}
	core.Handle = cublas.NewStandardImplementation(cublas.WithContext(_ctx))
	if core.Handle.Err() != nil {
		fmt.Println("fail to NewStandardImplementation, err: ", err)
		return nil, err
	}

	if core.Input, err = NewGPUBuffer(_ctx, maxBatch*maxDimension*maxPrecision); err != nil {
		fmt.Println("fail to Input, err: ", err)
		return nil, err
	}

	if core.Output, err = NewGPUBuffer(_ctx, buffer.Size()/minDimension*maxBatch); err != nil {
		fmt.Println("fail to Output, err: ", err)
		return nil, err
	}
	return core, nil
}

func (c *Core) Do(job SearchJob) {
	c.Queue <- job
}

func (c *Core) Work(ctx context.Context) {
	var (
		ret struct {
			Result   [][]FeatureSearchResult
			Duration map[string]time.Duration
			Err      error
		}
	)

	cu.SetCurrentContext(_ctx.CUDAContext())

	for {
		select {
		case job := <-c.Queue:

			var (
				err error
			)
			ret.Duration = make(map[string]time.Duration, 0)
			version := atomic.LoadUintptr(&job.Block.Version)
			now := time.Now()
			if job.Block.Index != c.Index || version != c.Version {
				if err = c.Buffer.Copy(job.Block.Buffer); err != nil {
					ret.Err = ErrWriteInputBuffer
					job.RetChan <- ret
				}
				c.Index = job.Block.Index
				c.Version = version
			}
			ret.Duration["preload"] = time.Since(now)

			ret.Result, ret.Err, ret.Duration["sgemm"], ret.Duration["DtoH"] = c.search(job.Block, job.Input, c.Output, job.Batch, job.Limit)
			job.RetChan <- ret
			//fmt.Println("Index: ", job.Block.Index, ", Duration:", duration)
		case <-ctx.Done():
			return
		}
	}
}

func (c *Core) search(block *Block, inputBuffer, outputBuffer Buffer, batch, limit int) (ret [][]FeatureSearchResult, err error, t1, t2 time.Duration) {
	height := block.NextIndex
	if height == 0 {
		return
	}

	//vec3 := make([]float32, height*batch)
	var vec FeatureValue
	var alpha, beta float32
	alpha = 1.0
	beta = 0.0
	now := time.Now()
	c.Handle.Sgemm(
		blas.Trans,
		blas.NoTrans,
		height,
		batch,
		block.Dims,
		alpha,
		*(*[]float32)(unsafe.Pointer(&reflect.SliceHeader{uintptr(c.Buffer.GetBuffer().(cu.DevicePtr)), c.Buffer.Size() / 4, c.Buffer.Size() / 4})),
		block.Dims,
		*(*[]float32)(unsafe.Pointer(&reflect.SliceHeader{uintptr(inputBuffer.GetBuffer().(cu.DevicePtr)), inputBuffer.Size() / 4, inputBuffer.Size() / 4})),
		block.Dims,
		beta,
		*(*[]float32)(unsafe.Pointer(&reflect.SliceHeader{uintptr(outputBuffer.GetBuffer().(cu.DevicePtr)), outputBuffer.Size() / 4, outputBuffer.Size() / 4})),
		height,
	)
	if err = c.Handle.Err(); err != nil {
		return
	}
	t1 = time.Since(now)
	now = time.Now()
	vec, err = outputBuffer.Read()
	if err != nil {
		return nil, errors.New("fail to read buffer vec3, err:" + err.Error()), t1, t2
	}
	t2 = time.Since(now)
	vec3 := *(*[]float32)(unsafe.Pointer(&reflect.SliceHeader{uintptr(unsafe.Pointer(&vec[0])), len(vec) / 4, len(vec) / 4}))
	for i := 0; i < batch; i++ {
		slc, err := outputBuffer.Slice(i*height*block.Precision, (i+1)*height*block.Precision)
		if slc == nil || err != nil {
			return nil, ErrSliceBuffer, t1, t2
		}
	}

	//  Trans result
	for i := 0; i < batch; i++ {
		var result []FeatureSearchResult
		indexes, scores := MaxNFloat32(vec3[i*height:(i+1)*height], limit)
		ids := block.GetIDs(indexes...)
		for j, _ := range indexes {
			if ids[j] != "" {
				r := FeatureSearchResult{Score: FeatureScore(scores[j]), ID: ids[j]}
				result = append(result, r)
			}
		}
		ret = append(ret, result)
	}

	return
}

func loadFeatures(features ...FeatureValue) (Buffer, error) {

	var (
		size   int
		target FeatureValue
		err    error
		buffer Buffer
	)

	for _, feature := range features {
		size += len(feature)
	}

	cu.SetCurrentContext(_ctx.CUDAContext())

	if buffer, err = NewGPUBuffer(_ctx, size); err != nil {
		return nil, err
	}

	for _, feature := range features {
		target = append(target, feature...)
	}
	if err = buffer.Write(target); err != nil {
		return nil, err
	}
	return buffer, nil
}
