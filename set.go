// +build cublas

package goFeature

import (
	"context"
	"sync"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

type SearchJob struct {
	Block
	Features []FeatureValue
	Batch    int
	Limit    int
	RetChan  chan struct {
		Result [][]FeatureSearchResult
		Err    error
	}
}

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
	InputBuffer     []Buffer
	OutputBuffer    []Buffer
	SearchQueue     chan SearchJob
	SearchLock      sync.Mutex
}

func (s *FeatureSet) doSearch(ctx context.Context, inputBuffer, outputBuffer Buffer) {

	var (
		ret struct {
			Result [][]FeatureSearchResult
			Err    error
		}
	)

	for {
		select {
		case job := <-s.SearchQueue:

			var (
				target FeatureValue
				err    error
			)
			target, err = FeatureValueTranspose1D(s.Precision, job.Features...)
			if err != nil {
				ret.Err = err
				job.RetChan <- ret
			}

			if err = inputBuffer.Write(target); err != nil {
				ret.Err = ErrWriteInputBuffer
				job.RetChan <- ret
			}

			ret.Result, ret.Err = job.Block.Search(inputBuffer, outputBuffer, job.Batch, job.Limit)
			job.RetChan <- ret
		case <-ctx.Done():
			return
		}
	}

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
			block.Accquire(s.Handle, s.Name, s.Dimension, s.Precision, s.Batch, s.doSearch)
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
			if err = block.Insert(feautres[offset:(offset + length)]...); err != nil {
				return
			}
			offset += length
			remain -= length
		}
		if remain <= 0 {
			break
		}
	}
	return
}

func (s *FeatureSet) Search(threshold FeatureScore, limit int, features ...FeatureValue) (ret [][]FeatureSearchResult, err error) {
	batch := len(features)
	if batch > s.Batch {
		return nil, ErrOutOfBatch
	}

	results := make([][]FeatureSearchResult, batch)
	retChan := make(chan struct {
		Result [][]FeatureSearchResult
		Err    error
	}, len(s.Blocks))
	for _, block := range s.Blocks {
		s.SearchQueue <- SearchJob{
			Block:    block,
			Features: features,
			Batch:    batch,
			Limit:    limit,
			RetChan:  retChan,
		}
	}
	for _, _ = range s.Blocks {
		r := <-retChan
		if r.Err != nil {
			return nil, r.Err
		}
		for b, r := range r.Result {
			var rr []FeatureSearchResult
			for _, r1 := range r {
				if r1.Score >= threshold {
					rr = append(rr, r1)
				}
			}
			results[b] = append(results[b], rr...)
		}
	}
	close(retChan)
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

	close(s.SearchQueue)

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
