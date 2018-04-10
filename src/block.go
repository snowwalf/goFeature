// +build cublas

package goFeature

import (
	"errors"
	"sync"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

type Block struct {
	// block info
	Index     int
	BlockSize int
	Buffer    cuda.Buffer
	Mutex     sync.Mutex
	Ctx       *cuda.Context

	// feature info
	Dims      int
	Precision int
	Owner     string
	Empty     []int
	NextIndex int
	IDs       []string
}

func NewBlock(ctx *cuda.Context, index, blockSize int, buffer cuda.Buffer) *Block {
	block := &Block{
		Index:     index,
		BlockSize: blockSize,
		Buffer:    buffer,
		Ctx:       ctx,
	}
	return block
}

func (b *Block) Capacity() int {
	length := b.BlockSize / (b.Precision * b.Dims)
	return len(b.Empty) + (length - b.NextIndex)
}

func (b *Block) Accquire(owner string, dims, precision int) {
	b.Dims = dims
	b.Precision = precision
	b.Owner = owner
	b.IDs = make([]string, b.BlockSize/(precision*dims))
}

func (b *Block) Insert(features []Feature) (err error) {
	if len(features) > b.Capacity() {
		return ErrBlockIsFull
	}

	vector := Float32FeautureTO1D(features)

	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	if len(b.Empty) > 0 {
		length := len(b.Empty)
		if len(features) < len(b.Empty) {
			length = len(features)
		}
		for i, index := range b.Empty[:length] {
			buffer := cuda.Slice(b.Buffer, uintptr(index*b.Dims*b.Precision), uintptr((index+1)*b.Dims*b.Precision))
			err = <-b.Ctx.Run(func() (e error) {
				e = cuda.WriteBuffer(buffer, vector[i*b.Dims:(i+1)*b.Dims])
				if e != nil {
					return ErrWriteCudaBuffer
				}
				return
			})
			b.IDs[index] = features[i].ID
		}
	}

	if len(features) > len(b.Empty) {
		buffer := cuda.Slice(b.Buffer, uintptr(b.NextIndex*b.Dims*b.Precision), uintptr((b.NextIndex+len(features)-len(b.Empty))*b.Dims*b.Precision))
		err = <-b.Ctx.Run(func() (e error) {
			e = cuda.WriteBuffer(buffer, vector[len(b.Empty)*b.Dims:])
			if e != nil {
				return ErrWriteCudaBuffer
			}
			return
		})

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

// Search :
//	search N features(s) in the block
//	empty block will return score=0 result
// Inputs :
// 	handle - cublash handle
//	inputbuffer - cuda buffer with  target feature(s)
//	outputbuffer - cuda buffer to store search result temporarily
//	batch - number of target feature(s)
// Output :
//	ret - search result with id and score
//  err - search error, or nil if success
func (b *Block) Search(handle *cublas.Handle, inputBuffer, outputBuffer cuda.Buffer, batch, limit int) (ret []SearchResult, err error) {
	dimension := b.Dims
	height := b.NextIndex
	if height == 0 {
		ret = make([]SearchResult, batch)
		return
	}

	vec3 := make([]float32, height*batch)
	err = <-b.Ctx.Run(func() (e error) {
		var alpha, beta float32
		alpha = 1.0
		beta = 0.0
		e = handle.Sgemm(
			cublas.Trans,
			cublas.NoTrans,
			height,
			batch,
			dimension,
			&alpha,
			b.Buffer,
			dimension,
			inputBuffer,
			dimension,
			&beta,
			outputBuffer,
			height,
		)
		if e != nil {
			return
		}
		e = cuda.ReadBuffer(vec3, outputBuffer)
		if e != nil {
			return errors.New("fail to read buffer vec3, err:" + e.Error())
		}
		for i := 0; i < batch; i++ {
			slc := cuda.Slice(outputBuffer, uintptr(i*height*b.Precision), uintptr((i+1)*height*b.Precision))
			if slc == nil {
				err = ErrSliceBuffer
				return
			}
		}
		return nil
	})
	if err != nil {
		return
	}
	for i := 0; i < batch; i++ {
		var result SearchResult
		indexes, scores := MaxNFloat32(vec3[i*height:(i+1)*height], limit)
		for j, index := range indexes {
			r := FeatureResult{Score: scores[j], ID: b.IDs[index]}
			result.Results = append(result.Results, r)
		}
		//result.Index, result.Score = b.IDs[index], score
		ret = append(ret, result)
	}

	return
}

// Delete :
// 	delete N feature(s) from block
// Input :
//	ids - features id(s) to be deleted
// Output:
//	deleted - ids which has been deleted from the block, id(s) of features which are not found in the block will be ignored
//	err - delete error, or nil if success
func (b *Block) Delete(ids []string) (deleted []string, err error) {
	targets := make(map[string]int, 0)
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
			buffer := cuda.Slice(b.Buffer, uintptr(index*b.Dims*b.Precision), uintptr((index+1)*b.Dims*b.Precision))
			err = <-b.Ctx.Run(func() (e error) {
				e = cuda.ClearBuffer(buffer)
				if e != nil {
					return ErrClearCudaBuffer
				}
				return
			})
			b.IDs[index] = ""
			b.Empty = append(b.Empty, index)
			deleted = append(deleted, id)
		}
	}
	return
}

// Update :
// 	update N feature(s) in the block
func (b *Block) Update(features []Feature) (err error) {
	return
}

// Destroy :
// 	destroy the whole block and clear the cuda memory
// Input :
// Output :
// 	err : destroy err, or nil if success
func (b *Block) Destroy() (err error) {
	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	err = <-b.Ctx.Run(func() (e error) {
		e = cuda.ClearBuffer(b.Buffer)
		if e != nil {
			return ErrClearCudaBuffer
		}
		return
	})

	b.Dims = 0
	b.Precision = 0
	b.Owner = ""
	b.IDs = make([]string, 0)
	b.Empty = make([]int, 0)
	b.NextIndex = 0

	return
}
