package goFeature

import (
	"sync"
)

type Set struct {
	*Cache
	Name            string
	Dimension       int
	BlockFeatureNum int
	Precision       int
	Batch           int
	Blocks          []*Block
	SearchQueue     chan SearchJob
	Mutex           sync.Mutex
}

// Add : interface of set
// Add: add features to set
// 	- features: features to be added
func (s *Set) Add(features ...Feature) (err error) {
	var empty int

	s.Mutex.Lock()
	defer s.Mutex.Unlock()
	for _, block := range s.Blocks {
		empty += block.Margin()
	}

	if len(features) > empty {
		remain := len(features) - empty
		blockLength := s.Cache.GetBlockSize() / (s.Dimension * s.Precision)
		blockNum := (remain + blockLength - 1) / blockLength
		var blocks []*Block
		if blocks, err = s.Cache.GetEmptyBlock(blockNum); err != nil {
			return
		}
		for _, block := range blocks {
			block.Accquire(s.Name, s.Dimension, s.Precision, s.Batch)
		}
		s.Blocks = append(s.Blocks, blocks...)
	}
	offset := 0
	remain := len(features)
	for _, block := range s.Blocks {
		length := block.Margin()
		if length > remain {
			length = remain
		}
		if length > 0 {
			if err = block.Insert(features[offset:(offset + length)]...); err != nil {
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

// Delete : try to delete features by id
//  - ids: feature id to be deleted
//  - deleted: feature ids deleted after function calling, nil if no one found
func (s *Set) Delete(ids ...FeatureID) (deleted []FeatureID, err error) {
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

// Update : try to update features
//  - features: feature to be deleted
//  - updated: feature ids updated after function calling, nil if no one found
func (s *Set) Update(features ...Feature) (updated []FeatureID, err error) {
	// TODO
	return
}

// Destroy :
// 	destroy the whole feature set and release resource
func (s *Set) Destroy() (err error) {
	s.Mutex.Lock()
	defer s.Mutex.Unlock()

	close(s.SearchQueue)

	for _, block := range s.Blocks {
		if err = block.Release(); err != nil {
			return
		}
	}
	return
}

// Read: try to read features by id
//  - ids: feature id to be read
//  - features: feature got after function calling, nil if no one found
func (s *Set) Read(ids ...FeatureID) ([]Feature, error) {
	// TODO
	return nil, nil
}
