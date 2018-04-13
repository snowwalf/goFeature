// +build cublas

package goFeature

import (
	"sync"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

type FeatureSet struct {
	*cuda.Context
	*cublas.Handle
	Name            string
	Dimension       int
	BlockFeatureNum int
	Precision       int
	Batch           int
	Blocks          []Block
	Cache           Cache
	InputBuffer     Buffer
	OutputBuffer    Buffer
	SearchLock      sync.Mutex
}

func (s *FeatureSet) Add(feautres ...Feature) (err error) {
	var empty int

	for _, block := range s.Blocks {
		empty += block.Margin()
	}

	if len(feautres) > empty {
		remain := len(feautres) - empty
		blockLength := s.Cache.GetBlockSize() / (s.Dimension * s.Precision)
		blockNum := (remain + blockLength - 1) / blockLength
		var blocks []Block
		if blocks, err = s.Cache.GetEmptyBlock(blockNum); err != nil {
			return
		}
		for _, block := range blocks {
			block.Accquire(s.Handle, s.Name, s.Dimension, s.Precision)
		}
		s.Blocks = append(s.Blocks, blocks...)
	}
	offset := 0
	remain := len(feautres)
	for _, block := range s.Blocks {
		length := block.Margin()
		if length > remain {
			length = remain
		}
		if length > 0 {
			block.Insert(feautres[offset:(offset + length)]...)
			offset += length
			remain -= length
		}
		if remain <= 0 {
			break
		}
	}
	return
}

func (s *FeatureSet) Search(threshold float32, limit int, features ...FeatureValue) (ret [][]FeatureSearchResult, err error) {
	batch := len(features)
	if batch > s.Batch {
		return nil, ErrOutOfBatch
	}
	var target FeatureValue
	target, err = FeatureValueTranspose1D(s.Precision, features...)
	if err != nil {
		return
	}
	s.SearchLock.Lock()
	defer s.SearchLock.Unlock()
	if err = s.InputBuffer.Write(target); err != nil {
		return nil, ErrWriteInputBuffer
	}

	results := make([][]FeatureSearchResult, batch)
	for _, block := range s.Blocks {
		result, e := block.Search(s.InputBuffer, s.OutputBuffer, batch, limit)
		if e != nil {
			return nil, e
		}
		for b, r := range result {
			results[b] = append(results[b], r...)
		}
	}
	for _, result := range results {
		_, features := MaxNFeatureResult(result, limit)
		ret = append(ret, features)
	}
	return
}

// Delete :
// 	delete N feature(s) from set
func (s *FeatureSet) Delete(ids ...FeatureID) (deleted []FeatureID, err error) {
	var del []FeatureID
	for _, block := range s.Blocks {
		if len(ids) == 0 {
			break
		}
		if del, err = block.Delete(ids...); err != nil {
			return
		}
		if len(del) == 0 {
			continue
		}
		var remain []FeatureID
		remain = append(remain, ids...)
		ids = make([]FeatureID, 0)
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
	return
}

func (s *FeatureSet) Update(features ...Feature) (updated []FeatureID, err error) {
	// TODO
	return
}

// Destroy :
// 	destroy the whole feature set and release resource
func (s *FeatureSet) Destroy() (err error) {
	s.SearchLock.Lock()
	defer s.SearchLock.Unlock()

	for _, block := range s.Blocks {
		if err = block.Release(); err != nil {
			return
		}
	}
	return
}

func (s *FeatureSet) Read(ids ...FeatureID) ([]Feature, error) {
	// TODO
	return nil, nil
}
