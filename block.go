// +build cublas

package goFeature

import (
	"errors"
	"sync"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

type _Block struct {
	// block info
	Index     int
	BlockSize int
	Buffer    Buffer
	Mutex     sync.Mutex
	Ctx       *cuda.Context
	Handle    *cublas.Handle

	// feature info
	Dims      int
	Precision int
	Owner     string
	Empty     []int
	NextIndex int
	IDs       []FeatureID
}

func NewBlock(ctx *cuda.Context, index, blockSize int, buffer Buffer) Block {
	block := &_Block{
		Index:     index,
		BlockSize: blockSize,
		Buffer:    buffer,
		Ctx:       ctx,
	}
	return block
}

func (b *_Block) IsOwned() bool { return b.Owner != "" }

func (b *_Block) Capacity() int { return b.BlockSize / (b.Precision * b.Dims) }

func (b *_Block) Margin() int {
	length := b.BlockSize / (b.Precision * b.Dims)
	return len(b.Empty) + (length - b.NextIndex)
}

func (b *_Block) Accquire(handle *cublas.Handle, owner string, dims, precision int) (err error) {
	if b.Owner != "" {
		return ErrBlockUsed
	}
	b.Dims = dims
	b.Precision = precision
	b.Owner = owner
	b.Handle = handle
	b.IDs = make([]FeatureID, b.BlockSize/(precision*dims))
	return
}

func (b *_Block) Insert(features ...Feature) (err error) {
	if len(features) > b.Margin() {
		return ErrBlockIsFull
	}

	vector, err := TByte(FeatureToFeatureValue1D(features...))
	if err != nil {
		return
	}

	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	if len(b.Empty) > 0 {
		length := len(b.Empty)
		if len(features) < len(b.Empty) {
			length = len(features)
		}
		for i, index := range b.Empty[:length] {
			buffer, err := b.Buffer.Slice(index*b.Dims*b.Precision, (index+1)*b.Dims*b.Precision)
			if err != nil {
				return err
			}
			if err = buffer.Write(FeatureValue(vector[i*b.Dims : (i+1)*b.Dims])); err != nil {
				return ErrWriteCudaBuffer
			}
			b.IDs[index] = features[i].ID
		}
	}

	if len(features) > len(b.Empty) {
		buffer, err := b.Buffer.Slice(b.NextIndex*b.Dims*b.Precision, (b.NextIndex+len(features)-len(b.Empty))*b.Dims*b.Precision)
		if err != nil {
			return err
		}
		if err = buffer.Write(FeatureValue(vector[len(b.Empty)*b.Dims:])); err != nil {
			return err
		}

		for i, feature := range features[len(b.Empty):] {
			b.IDs[b.NextIndex+i] = feature.ID
		}
	}

	if len(features) > len(b.Empty) {
		b.NextIndex += (len(features) - len(b.Empty))
		b.Empty = make([]int, 0)
	} else {
		b.Empty = b.Empty[len(features):]
	}
	return
}

// Delete :
// 	delete N feature(s) from block
func (b *_Block) Delete(ids ...FeatureID) (deleted []FeatureID, err error) {
	targets := make(map[FeatureID]int, 0)
	for _, id := range ids {
		targets[id] = -1
	}
	for index, value := range b.IDs {
		if _, exist := targets[value]; exist {
			targets[value] = index
		}
	}
	if len(targets) == 0 {
		return
	}
	b.Mutex.Lock()
	defer b.Mutex.Unlock()
	for id, index := range targets {
		if index != -1 {
			buffer, err := b.Buffer.Slice(index*b.Dims*b.Precision, (index+1)*b.Dims*b.Precision)
			if err != nil {
				return nil, err
			}
			if err = buffer.Reset(); err != nil {
				return nil, ErrClearCudaBuffer
			}
			b.IDs[index] = ""
			b.Empty = append(b.Empty, index)
			deleted = append(deleted, id)
		}
	}
	return
}

// Update :
// 	update N feature(s) in the block
func (b *_Block) Update(features ...FeatureID) (updated []FeatureID, err error) {
	// TODO
	return
}

// Read :
//  get features detail info from block
func (b *_Block) Read(...FeatureID) (features []Feature, err error) {
	// TODO
	return
}

// Search :
//	search N features(s) in the block
//	empty block will return score=0 result
func (b *_Block) Search(inputBuffer, outputBuffer Buffer, batch, limit int) (ret [][]FeatureSearchResult, err error) {
	dimension := b.Dims
	height := b.NextIndex
	if height == 0 {
		return
	}

	vec3 := make([]float32, height*batch*b.Precision)
	err = <-b.Ctx.Run(func() (e error) {
		var alpha, beta float32
		alpha = 1.0
		beta = 0.0
		e = b.Handle.Sgemm(
			cublas.Trans,
			cublas.NoTrans,
			height,
			batch,
			dimension,
			&alpha,
			b.Buffer.GetBuffer().(cuda.Buffer),
			dimension,
			inputBuffer.GetBuffer().(cuda.Buffer),
			dimension,
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
			slc, err := outputBuffer.Slice(i*height*b.Precision, (i+1)*height*b.Precision)
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
	for i := 0; i < batch; i++ {
		var result []FeatureSearchResult
		indexes, scores := MaxNFloat32(vec3[i*height:(i+1)*height], limit)
		for j, index := range indexes {
			r := FeatureSearchResult{Score: FeatureScore(scores[j]), ID: b.IDs[index]}
			result = append(result, r)
		}
		//result.Index, result.Score = b.IDs[index], score
		ret = append(ret, result)
	}

	return
}

// Release :
// 	release the whole block and clear the memory
func (b *_Block) Release() (err error) {
	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	if err = b.Buffer.Reset(); err != nil {
		return ErrClearCudaBuffer
	}

	b.Dims = 0
	b.Precision = 0
	b.Owner = ""
	b.IDs = make([]FeatureID, 0)
	b.Empty = make([]int, 0)
	b.NextIndex = 0

	return
}
