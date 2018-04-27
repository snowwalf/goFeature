// +build cublas

package goFeature

import (
	"context"
	"errors"
	"strconv"
	"sync/atomic"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

const (
	defaultSearchQueueSize = 10000
)

var (
	_ctx       *cuda.Context
	_allocator cuda.Allocator
)

// initCuda: try to init cuda context and allocator gpu buffer
func initCuda(gpuID, gpuMemSize int) (Buffer, error) {
	devices, err := cuda.AllDevices()
	if err != nil {
		return nil, errors.New("fail to get all gpu device, due to" + err.Error())
	}

	if len(devices) < (gpuID - 1) {
		return nil, errors.New("fail to bind to invalid gpu, id is " + strconv.Itoa(gpuID))
	}

	if _ctx, err = cuda.NewContext(devices[gpuID], 100); err != nil {
		return nil, errors.New("fail to init cuda context, due to " + err.Error())
	}

	totalMem, err := devices[0].TotalMem()
	if err != nil {
		return nil, errors.New("fail to get the total mem of gpu " + strconv.Itoa(gpuID))
	}
	if float64(totalMem)*0.95 < float64(gpuMemSize) {
		return nil, errors.New("try to allocate too much gpu mem, only " + strconv.FormatUint(totalMem, 10) + "B*0.8 can be used")
	}

	_allocator = cuda.GCAllocator(cuda.NativeAllocator(_ctx), 0)

	return NewGPUBuffer(_ctx, _allocator, gpuMemSize)
}

type Core struct {
	*cublas.Handle
	Buffer
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
	if core.Handle, err = cublas.NewHandle(_ctx); err != nil {
		return nil, err
	}

	if core.Input, err = NewGPUBuffer(_ctx, _allocator, maxBatch*maxDimension*maxPrecision); err != nil {
		return nil, err
	}

	if core.Output, err = NewGPUBuffer(_ctx, _allocator, buffer.Size()/minDimension*maxBatch); err != nil {
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
			Result [][]FeatureSearchResult
			Err    error
		}
	)

	for {
		select {
		case job := <-c.Queue:

			var (
				err error
			)

			version := atomic.LoadUintptr(&job.Block.Version)
			if job.Block.Index != c.Index || version != c.Version {
				if err = c.Buffer.Copy(job.Block.Buffer); err != nil {
					ret.Err = ErrWriteInputBuffer
					job.RetChan <- ret
				}
				c.Index = job.Block.Index
				c.Version = version
			}

			ret.Result, ret.Err = c.search(job.Block, job.Input, c.Output, job.Batch, job.Limit)
			job.RetChan <- ret
		case <-ctx.Done():
			return
		}
	}
}

func (c *Core) search(block *Block, inputBuffer, outputBuffer Buffer, batch, limit int) (ret [][]FeatureSearchResult, err error) {
	height := block.NextIndex
	if height == 0 {
		return
	}

	vec3 := make([]float32, height*batch)
	err = <-_ctx.Run(func() (e error) {
		var alpha, beta float32
		alpha = 1.0
		beta = 0.0
		e = c.Handle.Sgemm(
			cublas.Trans,
			cublas.NoTrans,
			height,
			batch,
			block.Dims,
			&alpha,
			c.Buffer.GetBuffer().(cuda.Buffer),
			block.Dims,
			inputBuffer.GetBuffer().(cuda.Buffer),
			block.Dims,
			&beta,
			outputBuffer.GetBuffer().(cuda.Buffer),
			height,
		)
		if e != nil {
			return
		}
		e = cuda.ReadBuffer(vec3, outputBuffer.GetBuffer().(cuda.Buffer))
		if e != nil {
			return errors.New("fail to read buffer vec3, err:" + e.Error())
		}
		for i := 0; i < batch; i++ {
			slc, err := outputBuffer.Slice(i*height*block.Precision, (i+1)*height*block.Precision)
			if slc == nil || err != nil {
				e = ErrSliceBuffer
				return
			}
		}
		return nil
	})
	if err != nil {
		return
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

	if buffer, err = NewGPUBuffer(_ctx, _allocator, size); err != nil {
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
