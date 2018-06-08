package goFeature

import (
	"context"
	"sync"
	"time"
)

type _Manager struct {

	// Sets
	Sets  map[string]*Set
	Mutex sync.RWMutex

	Cores []*Core
	Cache *Cache
}

type SearchJob struct {
	*Block
	Input     Buffer
	Dimension int
	Precision int
	Batch     int
	Limit     int
	RetChan   chan struct {
		Result   [][]FeatureSearchResult
		Duration map[string]time.Duration
		Err      error
	}
}

const (
	defaultBlockSize = 32 * 1024 * 1024
	maxDimension     = 4096
	minDimension     = 256
	maxPrecision     = 8
	maxBatch         = 16
)

var _ Interface = &_Manager{}

func NewManager(ctx context.Context, gpuID, gpuMem, blockNum, blockSize int) (*_Manager, error) {
	mgr := &_Manager{
		Sets: make(map[string]*Set, 0),
	}

	// Init cores
	if gpuMem > 0 {
		buffer, err := initCuda(gpuID, gpuMem)
		if err != nil {
			return nil, err
		}

		for i := 0; i < gpuMem/blockSize; i++ {
			buf, err := buffer.Slice(i*blockSize, (i+1)*blockSize)
			if err != nil {
				return nil, err
			}
			core, err := NewCore(buf)
			if err != nil {
				return nil, err
			}
			mgr.Cores = append(mgr.Cores, core)
			go core.Work(ctx)
		}
	}

	// Init Cache
	mgr.Cache = NewCache(blockNum, blockSize)

	return mgr, nil
}

func (m *_Manager) NewSet(name string, dims, precision, batch int) (err error) {
	m.Mutex.RLock()
	if _, exist := m.Sets[name]; exist {
		m.Mutex.RUnlock()
		return ErrFeatureSetExist
	}
	m.Mutex.RUnlock()

	set := &Set{
		Dimension:       dims,
		Precision:       precision,
		BlockFeatureNum: m.Cache.GetBlockSize() / (dims * precision),
		Name:            name,
		Batch:           batch,
		Cache:           m.Cache,
		SearchQueue:     make(chan SearchJob, 10),
	}

	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	m.Sets[name] = set
	return
}

func (m *_Manager) DestroySet(name string) (err error) {
	m.Mutex.Lock()
	set, exist := m.Sets[name]
	if !exist {
		m.Mutex.Unlock()
		return ErrFeatureSetNotFound
	}
	m.Mutex.Unlock()

	if err = set.Destroy(); err != nil {
		return
	}
	m.Mutex.Lock()
	defer m.Mutex.Unlock()
	delete(m.Sets, name)
	return
}

func (m *_Manager) set(name string) (set *Set, err error) {
	m.Mutex.RLock()
	defer m.Mutex.RUnlock()
	set, exist := m.Sets[name]
	if !exist {
		return nil, ErrFeatureSetNotFound
	}
	return
}

func (m *_Manager) GetSet(name string) (dimesion, precision, batch int, err error) {
	set, err := m.set(name)
	if err != nil {
		return
	}

	return set.Dimension, set.Precision, set.Batch, nil
}

func (m *_Manager) UpdateSet(name string, batch int) (err error) {
	if batch > maxBatch {
		return ErrBatchTooLarge
	}
	set, err := m.set(name)
	if err != nil {
		return
	}
	set.Batch = batch
	return nil
}

func (m *_Manager) AddFeature(name string, features ...Feature) (err error) {
	set, err := m.set(name)
	if err != nil {
		return
	}

	return set.Add(features...)
}

func (m *_Manager) DeleteFeature(name string, ids ...FeatureID) (deleted []FeatureID, err error) {
	set, err := m.set(name)
	if err != nil {
		return
	}

	return set.Delete(ids...)
}

func (m *_Manager) UpdateFeature(name string, features ...Feature) (updated []FeatureID, err error) {
	set, err := m.set(name)
	if err != nil {
		return
	}

	return set.Update(features...)
}

func (m *_Manager) ReadFeautre(name string, ids ...FeatureID) (features []Feature, err error) {
	set, err := m.set(name)
	if err != nil {
		return
	}

	return set.Read(ids...)
}

func (m *_Manager) Search(name string, threshold FeatureScore, limit int, features ...FeatureValue) (ret [][]FeatureSearchResult, err error) {
	set, err := m.set(name)
	if err != nil {
		return
	}
	duration := map[string]time.Duration{"preload": 0, "sgemm": 0, "DtoH": 0}
	batch := len(features)
	if batch > set.Batch {
		return nil, ErrOutOfBatch
	}

	input, err := loadFeatures(features...)
	if err != nil {
		return
	}

	results := make([][]FeatureSearchResult, batch)
	retChan := make(chan struct {
		Result   [][]FeatureSearchResult
		Duration map[string]time.Duration
		Err      error
	}, len(set.Blocks))
	for _, block := range set.Blocks {
		core := m.Cores[block.Index%len(m.Cores)]
		core.Do(SearchJob{
			Block:   block,
			Input:   input,
			Batch:   batch,
			Limit:   limit,
			RetChan: retChan,
		})

	}
	for _ = range set.Blocks {
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
		for index, du := range r.Duration {
			duration[index] += du
		}
	}
	close(retChan)
	for _, result := range results {
		_, features := MaxNFeatureResult(result, limit)
		ret = append(ret, features)
	}

	return
}
