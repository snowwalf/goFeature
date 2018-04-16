// +build cublas

package goFeature

import (
	"sync"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

type _Cache struct {
	Ctx       *cuda.Context
	AllBlocks []Block
	Allocator cuda.Allocator
	BlockSize int
	Mutex     sync.Mutex
	Sets      map[string]Set
}

func NewCache(ctx *cuda.Context, blockNum, blockSize int) (cache *_Cache, err error) {
	cache = &_Cache{
		Ctx:       ctx,
		BlockSize: blockSize,
		Sets:      make(map[string]Set, 0),
		Allocator: cuda.GCAllocator(cuda.NativeAllocator(ctx), 0),
	}
	var buffer Buffer
	buffer, err = NewGPUBuffer(ctx, cache.Allocator, blockNum*blockSize)
	if err != nil {
		return
	}
	for i := 0; i < blockNum; i++ {
		slc, e := buffer.Slice(i*blockSize, (i+1)*blockSize)
		if e != nil || slc == nil {
			err = ErrSliceGPUBuffer
			return
		}
		block := NewBlock(ctx, cache.Allocator, i, blockSize, slc)
		cache.AllBlocks = append(cache.AllBlocks, block)
	}
	return
}

func (c *_Cache) NewSet(name string, dims, precision, batch int) (err error) {
	c.Mutex.Lock()
	if _, exist := c.Sets[name]; exist {
		c.Mutex.Unlock()
		return ErrFeatureSetExist
	}
	c.Mutex.Unlock()

	set := &FeatureSet{
		Context:         c.Ctx,
		Dimension:       dims,
		Precision:       precision,
		BlockFeatureNum: c.BlockSize / (dims * precision),
		Name:            name,
		Batch:           batch,
		Cache:           c,
		SearchQueue:     make(chan SearchJob, 10),
	}

	if set.Handle, err = cublas.NewHandle(c.Ctx); err != nil {
		return
	}

	c.Mutex.Lock()
	defer c.Mutex.Unlock()
	c.Sets[name] = set
	return
}

func (c *_Cache) DestroySet(name string) (err error) {
	c.Mutex.Lock()
	set, exist := c.Sets[name]
	if !exist {
		c.Mutex.Unlock()
		return ErrFeatureSetNotFound
	}
	c.Mutex.Unlock()

	if err = set.Destroy(); err != nil {
		return
	}
	c.Mutex.Lock()
	defer c.Mutex.Unlock()
	delete(c.Sets, name)
	return
}

func (c *_Cache) GetSet(name string) (set Set, err error) {
	c.Mutex.Lock()
	defer c.Mutex.Unlock()
	set, exist := c.Sets[name]
	if !exist {
		return nil, ErrFeatureSetNotFound
	}
	return
}

func (c *_Cache) GetBlockSize() int { return c.BlockSize }

func (c *_Cache) GetEmptyBlock(blockNum int) ([]Block, error) {
	var emptyBlocks []Block
	for _, block := range c.AllBlocks {
		if !block.IsOwned() {
			emptyBlocks = append(emptyBlocks, block)
		}
	}
	if len(emptyBlocks) < blockNum {
		return nil, ErrNotEnoughBlocks
	}
	return emptyBlocks[:blockNum], nil
}
