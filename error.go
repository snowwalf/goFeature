// +build cublas

package goFeature

import (
	"errors"
)

var (
	// buffer error
	ErrBufferWriteOutofRange = errors.New("try to write too much data into buffer")
	ErrBufferCopyOutofRange  = errors.New("try to copy too much data from source buffer")
	ErrBufferSliceOutofRange = errors.New("slice buffer out of range")
	ErrInvalidBufferData     = errors.New("invalid buffer data")
	ErrAllocateGPUBuffer     = errors.New("failed to allocate gpu buffer")

	// server error
	ErrInvalidDeviceID    = errors.New("invalid device id")
	ErrInvalidFeautres    = errors.New("invalid features in request")
	ErrFeatureSetNotFound = errors.New("feature set not found")
	ErrInvalidSetState    = errors.New("invalid set state")

	// cache error
	ErrFeatureSetExist  = errors.New("feature set is already exist")
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
	ErrBlockUsed   = errors.New("block is used")

	// cuda error
	ErrWriteCudaBuffer = errors.New("write to cuda buffer error")
	ErrSliceBuffer     = errors.New("fail to slice cuda buffer")
	ErrClearCudaBuffer = errors.New("clear cuda buffer failed")
	ErrAllDevice       = errors.New("fail to list all cuda device")
	ErrCreateContext   = errors.New("fail to create cuda context")

	// feature error
	ErrBadTransposeValue = errors.New("invalid transpose value to transpose")
)
