// +build cublas

package goFeature

import "github.com/unixpickle/cuda"

// GPU memory buffer
type GPUBuffer struct {
	cuda.Buffer
	*cuda.Context
}

var _ Buffer = &GPUBuffer{}

func NewGPUBuffer(ctx *cuda.Context, allocator cuda.Allocator, size int) (*GPUBuffer, error) {
	buffer := &GPUBuffer{
		Context: ctx,
	}
	err := <-ctx.Run(func() (e error) {
		buffer.Buffer, e = cuda.AllocBuffer(allocator, uintptr(size))
		if e != nil {
			return ErrAllocateGPUBuffer
		}
		if e = cuda.ClearBuffer(buffer.Buffer); e != nil {
			return
		}
		return nil
	})
	return buffer, err
}

func (b *GPUBuffer) GetBuffer() interface{} { return b.Buffer }

func (b *GPUBuffer) Write(value FeatureValue) (err error) {
	if len(value) > b.Size() {
		err = ErrBufferWriteOutofRange
		return
	}
	err = <-b.Context.Run(func() (e error) {
		return cuda.WriteBuffer(b.Buffer, []byte(value))
	})
	return

}

func (b *GPUBuffer) Read() (value FeatureValue, err error) {
	value = make(FeatureValue, b.Size())
	err = <-b.Context.Run(func() (e error) {
		return cuda.ReadBuffer([]byte(value), b.Buffer)
	})
	return
}

func (b *GPUBuffer) Copy(src Buffer) (err error) {
	if src.Size() > b.Size() {
		err = ErrBufferCopyOutofRange
	}
	err = <-b.Context.Run(func() (e error) {
		return cuda.CopyBuffer(b.Buffer, src.GetBuffer().(cuda.Buffer))
	})
	return
}

func (b *GPUBuffer) Size() int { return int(b.Buffer.Size()) }

func (b *GPUBuffer) Reset() (err error) {
	err = <-b.Context.Run(func() (e error) {
		return cuda.ClearBuffer(b.Buffer)
	})
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
		Buffer:  cuda.Slice(b.Buffer, uintptr(start), uintptr(end)),
		Context: b.Context,
	}, nil
}

// CPU memory buffer
type CPUBuffer struct {
	Buffer []byte
}

var _ Buffer = &CPUBuffer{}

func (b *CPUBuffer) GetBuffer() interface{} { return b.Buffer }

func (b *CPUBuffer) Write(value FeatureValue) (err error) {
	if len(value) > b.Size() {
		return ErrBufferWriteOutofRange
	}
	copy(b.Buffer, value)
	return nil
}

func (b *CPUBuffer) Read() (value FeatureValue, err error) {
	return b.Buffer, nil
}

func (b *CPUBuffer) Copy(src Buffer) error {
	copy(b.Buffer, src.GetBuffer().([]byte))
	return nil
}

func (b *CPUBuffer) Reset() error {
	for i := 0; i < b.Size(); i++ {
		b.Buffer[i] = byte(0)
	}
	return nil
}

func (b *CPUBuffer) Slice(start, end int) (buf Buffer, err error) {
	if start < 0 || start > b.Size() {
		return nil, ErrBufferSliceOutofRange
	}
	if end < 0 || end > b.Size() || end < start {
		return nil, ErrBufferSliceOutofRange
	}
	return &CPUBuffer{
		Buffer: b.Buffer[start:end],
	}, nil
}

func (b *CPUBuffer) Size() int { return len(b.Buffer) }
