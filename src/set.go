// +build cublas

package goFeature

import (
	"errors"
	"sync"
	"sync/atomic"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

type FeatureSet struct {
	Ctx             *cuda.Context
	Name            string
	Dimension       int
	BlockFeatureNum int
	Precision       int
	Batch           int
	Version         uint64
	State           uint32
	Blocks          []*Block
	Cache           *Cache
	InputBuffer     cuda.Buffer
	OutputBuffer    cuda.Buffer
	SearchLock      sync.Mutex
}

// set working state
const (
	SetStateUnknown int = iota
	SetStateCreated
	SetStateInitialized
)

type Feature struct {
	Value []float32 `json:"value"`
	ID    string    `json:"id,omitempty"`
}

type FeatureResult struct {
	Score float32 `json:"score"`
	ID    string  `json:"id,omitempty"`
}

type SearchResult struct {
	Results []FeatureResult `json:"results"`
}

func NewFeatureSet(cache *Cache, name string, dims, precision, batch, state int, version uint64) (set *FeatureSet, err error) {
	set = &FeatureSet{
		Ctx:             cache.Ctx,
		Dimension:       dims,
		Precision:       precision,
		BlockFeatureNum: cache.BlockSize / (dims * precision),
		Name:            name,
		Batch:           batch,
		Cache:           cache,
		Version:         version,
		State:           uint32(state),
	}

	if set.InputBuffer, err = cache.AccquireBuffer(batch * dims * precision); err != nil {
		return nil, err
	}
	if set.OutputBuffer, err = cache.AccquireBuffer(batch * set.BlockFeatureNum * precision); err != nil {
		return nil, err
	}
	return
}

func (s *FeatureSet) UpdateState(state int) { atomic.StoreUint32(&s.State, uint32(state)) }

func (s *FeatureSet) Add(blockSize int, feautres []Feature) (err error) {
	var empty int

	for _, block := range s.Blocks {
		empty += block.Capacity()
	}

	if len(feautres) > empty {
		remain := len(feautres) - empty
		blockLength := blockSize / (s.Dimension * s.Precision)
		blockNum := (remain + blockLength - 1) / blockLength
		var blocks []*Block
		if blocks, err = s.Cache.GetEmptyBlock(blockNum); err != nil {
			return
		}
		for _, block := range blocks {
			block.Accquire(s.Name, s.Dimension, s.Precision)
		}
		s.Blocks = append(s.Blocks, blocks...)
	}
	offset := 0
	remain := len(feautres)
	for _, block := range s.Blocks {
		length := block.Capacity()
		if length > remain {
			length = remain
		}
		if length > 0 {
			block.Insert(feautres[offset:(offset + length)])
			offset += length
			remain -= length
		}
		if remain <= 0 {
			break
		}
	}
	if atomic.LoadUint32(&s.State) == uint32(SetStateInitialized) {
		atomic.AddUint64(&s.Version, 1)
	}
	return
}

func (s *FeatureSet) Search(handle *cublas.Handle, features [][]float32, limit int) (ret []SearchResult, err error) {
	batch := len(features)
	if batch > s.Batch {
		return nil, ErrOutOfBatch
	}
	target := TFloat32(features, s.Dimension, batch)
	s.SearchLock.Lock()
	defer s.SearchLock.Unlock()
	err = <-s.Ctx.Run(func() (e error) {
		e = cuda.WriteBuffer(s.InputBuffer, target)
		if e != nil {
			return errors.New("fail to write input buffer target, err:" + err.Error())
		}
		return
	})
	if err != nil {
		return nil, ErrWriteInputBuffer
	}

	results := make([][]FeatureResult, batch)
	for _, block := range s.Blocks {
		result, e := block.Search(handle, s.InputBuffer, s.OutputBuffer, batch, limit)
		if e != nil {
			return nil, e
		}
		for b, r := range result {
			results[b] = append(results[b], r.Results...)
		}
	}
	for _, result := range results {
		_, features := MaxNFeatureResult(result, limit)
		sr := SearchResult{Results: features}
		ret = append(ret, sr)
	}
	return
}

// Delete :
// 	delete N feature(s) from set
// Input :
//	ids - features id(s) to be deleted
// Output:
//	deleted - ids which has been deleted from the set, id(s) of features which are not found in the set will be ignored
//	err - delete error, or nil if success
func (s *FeatureSet) Delete(ids []string) (deleted []string, err error) {
	var del []string
	for _, block := range s.Blocks {
		if len(ids) == 0 {
			break
		}
		del, err = block.Delete(ids)
		if err != nil {
			return
		}
		if len(del) == 0 {
			continue
		}
		var remain []string
		remain = append(remain, ids...)
		ids = make([]string, 0)
		deleted = append(deleted, del...)
		for _, id := range remain {
			flag := false
			for _, d := range del {
				if id == d {
					flag = true
					break
				}
			}
			if !flag {
				ids = append(ids, id)
			}
		}
	}
	if atomic.LoadUint32(&s.State) == uint32(SetStateInitialized) {
		atomic.AddUint64(&s.Version, 1)
	}
	return
}

func (s *FeatureSet) Update(features []Feature) (err error) {
	if atomic.LoadUint32(&s.State) == uint32(SetStateInitialized) {
		atomic.AddUint64(&s.Version, 1)
	}
	return
}

// Destroy :
// 	destroy the whole feature set and release resource
// Input :
// Output :
// 	err - destroy err, or nil if success
func (s *FeatureSet) Destroy() (err error) {
	s.SearchLock.Lock()
	defer s.SearchLock.Unlock()

	for _, block := range s.Blocks {
		if err = block.Destroy(); err != nil {
			return
		}
	}
	return
}
