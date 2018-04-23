package goFeature

import (
	"sync"
)

type Cache struct {
	Blocks    []*Block
	BlockSize int
	Mutex     sync.Mutex
}

func NewCache(blockNum, blockSize int) *Cache {
	cache := &Cache{
		BlockSize: blockSize,
	}
	for i := 0; i < blockNum; i++ {
		block := NewBlock(i, blockSize)
		cache.Blocks = append(cache.Blocks, block)
	}
	return cache
}

// func (c *Cache) NewSet(name string, dims, precision, batch int) (err error) {
// 	c.Mutex.Lock()
// 	if _, exist := c.Sets[name]; exist {
// 		c.Mutex.Unlock()
// 		return ErrFeatureSetExist
// 	}
// 	c.Mutex.Unlock()

// 	set := &FeatureSet{
// 		Context:         c.Ctx,
// 		Dimension:       dims,
// 		Precision:       precision,
// 		BlockFeatureNum: c.BlockSize / (dims * precision),
// 		Name:            name,
// 		Batch:           batch,
// 		Cache:           c,
// 		SearchQueue:     make(chan SearchJob, 10),
// 	}

// 	if set.Handle, err = cublas.NewHandle(c.Ctx); err != nil {
// 		return
// 	}

// 	c.Mutex.Lock()
// 	defer c.Mutex.Unlock()
// 	c.Sets[name] = set
// 	return
// }

// func (c *Cache) DestroySet(name string) (err error) {
// 	c.Mutex.Lock()
// 	set, exist := c.Sets[name]
// 	if !exist {
// 		c.Mutex.Unlock()
// 		return ErrFeatureSetNotFound
// 	}
// 	c.Mutex.Unlock()

// 	if err = set.Destroy(); err != nil {
// 		return
// 	}
// 	c.Mutex.Lock()
// 	defer c.Mutex.Unlock()
// 	delete(c.Sets, name)
// 	return
// }

// func (c *Cache) GetSet(name string) (set Set, err error) {
// 	c.Mutex.Lock()
// 	defer c.Mutex.Unlock()
// 	set, exist := c.Sets[name]
// 	if !exist {
// 		return nil, ErrFeatureSetNotFound
// 	}
// 	return
// }

func (c *Cache) GetBlockSize() int { return c.BlockSize }

func (c *Cache) GetEmptyBlock(blockNum int) ([]*Block, error) {
	var emptyBlocks []*Block
	for _, block := range c.Blocks {
		if !block.IsOwned() {
			emptyBlocks = append(emptyBlocks, block)
		}
	}
	if len(emptyBlocks) < blockNum {
		return nil, ErrNotEnoughBlocks
	}
	return emptyBlocks[:blockNum], nil
}
