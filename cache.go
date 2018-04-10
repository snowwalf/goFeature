// +build cublas

package goFeature

import (
	"errors"
	"fmt"

	"github.com/unixpickle/cuda"
)

type Cache struct {
	Ctx         *cuda.Context
	AllBlocks   []*Block
	BlockSize   int
	FeatureSets map[string]FeatureSet
}

func (c *Cache) GetEmptyBlock(blockNum int) ([]*Block, error) {
	var emptyBlocks []*Block
	for _, block := range c.AllBlocks {
		if len(block.Owner) == 0 {
			emptyBlocks = append(emptyBlocks, block)
		}
	}
	if len(emptyBlocks) < blockNum {
		return nil, ErrNotEnoughBlocks
	}
	return emptyBlocks[:blockNum], nil
}

func (c *Cache) GetFeatureSets() map[string]FeatureSet {
	return c.FeatureSets
}

func NewCache(ctx *cuda.Context, blockNum, blockSize int) (cache *Cache, err error) {
	cache = &Cache{}
	allocator := cuda.GCAllocator(cuda.NativeAllocator(ctx), 0)
	var buffer cuda.Buffer
	err = <-ctx.Run(func() (err error) {
		buffer, err = cuda.AllocBuffer(allocator, uintptr(blockNum*blockSize))
		if err != nil {
			return errors.New("fail to allocate buffer " + fmt.Sprintf("%d*%d", blockNum, blockSize))
		}
		err = cuda.ClearBuffer(buffer)
		if err != nil {
			return errors.New("fail to clear buffer")
		}
		return nil
	})
	if err != nil {
		err = ErrAllocatGPUMemory
		return
	}
	for i := 0; i < blockNum; i++ {
		slc := cuda.Slice(buffer, uintptr(i*blockSize), uintptr((i+1)*blockSize))
		if slc == nil {
			err = ErrSliceGPUBuffer
			return
		}
		block := NewBlock(ctx, i, blockSize, slc)
		// block := Block{Index: i, BlockSize: blockSize, Length: 0, Buffer: slc, Dims: 0}
		cache.AllBlocks = append(cache.AllBlocks, block)
	}
	cache.BlockSize = blockSize
	cache.Ctx = ctx
	cache.FeatureSets = make(map[string]FeatureSet, 0)
	return
}

func (c *Cache) AccquireBuffer(size int) (buffer cuda.Buffer, err error) {
	allocator := cuda.GCAllocator(cuda.NativeAllocator(c.Ctx), 0)
	err = <-c.Ctx.Run(func() (err error) {
		buffer, err = cuda.AllocBuffer(allocator, uintptr(size))
		if err != nil {
			return errors.New("fail to allocate buffer " + fmt.Sprintf("%d", size))
		}
		return nil
	})
	if err != nil {
		err = ErrAllocatGPUMemory
		return
	}
	return
}
