package goFeature

// CPU memory buffer
type CPUBuffer struct {
	Buffer []byte
}

var _ Buffer = &CPUBuffer{}

func NewCPUBuffer(size int) *CPUBuffer {
	return &CPUBuffer{
		Buffer: make([]byte, size, size),
	}
}

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
