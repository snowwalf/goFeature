// +build cublas

package goFeature

import (
	"fmt"
	"unsafe"

	"gorgonia.org/cu"
)

// GPU memory buffer
type GPUBuffer struct {
	Ctx  *cu.Ctx
	Ptr  cu.DevicePtr
	size int
}

var _ Buffer = &GPUBuffer{}

func NewGPUBuffer(ctx *cu.Ctx, size int) (*GPUBuffer, error) {

	ptr, err := ctx.MemAlloc(int64(size))
	if err != nil {
		fmt.Errorf("NewGPUBuffer (%d) bytes, error: %s", size, err)
		return nil, ErrAllocateGPUBuffer
	}
	buffer := &GPUBuffer{
		Ctx:  ctx,
		Ptr:  ptr,
		size: size,
	}
	return buffer, err
}

func (b *GPUBuffer) GetBuffer() interface{} { return b.Ptr }

func (b *GPUBuffer) Write(value FeatureValue) (err error) {

	if len(value) > b.Size() {
		err = ErrBufferWriteOutofRange
		return
	}

	b.Ctx.MemcpyHtoD(cu.DevicePtr(b.Ptr), unsafe.Pointer(&value[0]), int64(len(value)))
	return
}

func (b *GPUBuffer) Read() (value FeatureValue, err error) {
	value = make(FeatureValue, b.Size())
	b.Ctx.MemcpyDtoH(unsafe.Pointer(&value[0]), cu.DevicePtr(b.Ptr), int64(b.Size()))
	return
}

func (b *GPUBuffer) Copy(src Buffer) (err error) {
	if src.Size() > b.Size() {
		err = ErrBufferCopyOutofRange
	}

	switch src.(type) {
	case *CPUBuffer:
		return b.Write(src.GetBuffer().([]byte))
	case *GPUBuffer:
		b.Ctx.Memcpy(cu.DevicePtr(b.Ptr), src.GetBuffer().(cu.DevicePtr), int64(src.Size()))
		return
	default:
		err = ErrInvalidBufferType
	}
	return
}

func (b *GPUBuffer) Size() int { return b.size }

func (b *GPUBuffer) Reset() (err error) {
	b.Ctx.MemsetD8(cu.DevicePtr(b.Ptr), 0, int64(b.Size()))
	return
}

func (b *GPUBuffer) Slice(start, end int) (buf Buffer, err error) {
	if start < 0 || start > b.Size() {
		return nil, ErrBufferSliceOutofRange
	}
	if end < 0 || end > b.Size() || end < start {
		return nil, ErrBufferSliceOutofRange
	}

	return &GPUBuffer{
		Ctx:  b.Ctx,
		Ptr:  b.Ptr + cu.DevicePtr(start),
		size: end - start,
	}, nil
}
