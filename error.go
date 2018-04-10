// +build cublas

package goFeature

import (
	"errors"
)

var (
	// server error
	ErrFeatureSetExist    = errors.New("feature set is already exist")
	ErrInvalidDeviceID    = errors.New("invalid device id")
	ErrInvalidFeautres    = errors.New("invalid features in request")
	ErrFeatureSetNotFound = errors.New("feature set not found")
	ErrInvalidSetState    = errors.New("invalid set state")

	// cache error
	ErrTooMuchGPUMemory = errors.New("use too much gpu memory")
	ErrAllocatGPUMemory = errors.New("fail to allocate gpu memory")
	ErrSliceGPUBuffer   = errors.New("fail to slice gpu buffer")
	ErrNotEnoughBlocks  = errors.New("cache does not have enough blocks")

	// feature set error
	ErrOutOfBatch        = errors.New("requests out of batch limit")
	ErrMismatchDimension = errors.New("feature with mismatch dimension")
	ErrWriteInputBuffer  = errors.New("failed to write input buffer")
	ErrWriteOutputBuffer = errors.New("failed to write output buffer")

	// block error
	ErrBlockIsFull = errors.New("block is full")

	// cuda error
	ErrWriteCudaBuffer = errors.New("write to cuda buffer error")
	ErrSliceBuffer     = errors.New("fail to slice cuda buffer")
	ErrClearCudaBuffer = errors.New("clear cuda buffer failed")
	ErrAllDevice       = errors.New("fail to list all cuda device")
	ErrCreateContext   = errors.New("fail to create cuda context")
)
