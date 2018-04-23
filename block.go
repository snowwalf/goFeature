package goFeature

import (
	"sync"
	"sync/atomic"
)

type Block struct {
	// block info
	Index     int
	Version   uintptr
	BlockSize int
	Buffer    Buffer
	Mutex     sync.Mutex

	// feature info
	Dims      int
	Precision int
	Batch     int
	Owner     string
	Empty     []int
	NextIndex int
	IDs       []FeatureID

	// internal
}

func NewBlock(index, blockSize int) *Block {
	block := &Block{
		Index:     index,
		BlockSize: blockSize,
		Buffer:    NewCPUBuffer(blockSize),
	}
	return block
}

// IsOwned : check if accquired
func (b *Block) IsOwned() bool { return b.Owner != "" }

// Capacity : get max feautre number of the block
func (b *Block) Capacity() int { return b.BlockSize / (b.Precision * b.Dims) }

// Margin : number of features can be inserted
func (b *Block) Margin() int {
	length := b.BlockSize / (b.Precision * b.Dims)
	return len(b.Empty) + (length - b.NextIndex)
}

// GetBuffer : get buffer in the block
func (b *Block) GetBuffer() Buffer { return b.Buffer }

// GetIDs : get feature id by the index
func (b *Block) GetIDs(indexes ...int) (ids []FeatureID) {
	for _, index := range indexes {
		var id FeatureID
		if index > 0 || index < b.NextIndex {
			id = b.IDs[index]
		}
		ids = append(ids, id)
	}
	return
}

// Accquire : one set tries to accquire the block
//  - owner: set name, unique
//  - dims: dimension of feature
//  - precision: precision of feature
//  - batch: batch limit of the set
func (b *Block) Accquire(owner string, dims, precision, batch int) (err error) {
	if b.Owner != "" {
		return ErrBlockUsed
	}
	b.Dims = dims
	b.Precision = precision
	b.Owner = owner
	b.Version = 0
	b.IDs = make([]FeatureID, b.BlockSize/(precision*dims))
	return
}

// Insert : insert features into the block
//  - features: features to be inserted
func (b *Block) Insert(features ...Feature) (err error) {
	if len(features) > b.Margin() {
		return ErrBlockIsFull
	}

	vector, err := TFeatureValue(FeatureToFeatureValue1D(features...))
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
	atomic.AddUintptr(&b.Version, 1)
	return
}

// Delete : try to delete features by id
//  - ids: feature id to be deleted
//  - deleted: feature ids deleted after function calling, nil if no one found
func (b *Block) Delete(ids ...FeatureID) (deleted []FeatureID, err error) {
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
	atomic.AddUintptr(&b.Version, 1)
	return
}

// Update : try to update features
//  - features: feature to be deleted
//  - updated: feature ids updated after function calling, nil if no one found
func (b *Block) Update(features ...Feature) (updated []FeatureID, err error) {
	// TODO
	atomic.AddUintptr(&b.Version, 1)
	return
}

// Read : try to read features by id
//  - ids: feature id to be read
//  - features: feature got after function calling, nil if no one found
func (b *Block) Read(ids ...FeatureID) (features []Feature, err error) {
	// TODO
	return
}

// Release : release the accquired block
func (b *Block) Release() (err error) {
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
	b.Version = 0

	return
}
