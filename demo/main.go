// +build cublas

package main

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	fs "github.com/snowwalf/goFeature"
	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

const (
	//BlockSize : size of feature block
	BlockSize = 1024 * 1024 * 32
	BlockNum  = 100
	Dimension = 512
	Batch     = 2
	Precision = 4
	Round     = 1
	SetSize   = 2
	SetNum    = 1
)

var (
	ctx  *cuda.Context
	sets []*fs.FeatureSet
)

func RandPickFeature(index int) (id string, err error) {
	if len(sets) < index {
		err = errors.New("RandPickFeature: index is out of bound")
		return
	}
	set := sets[index]
	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	bIndex := random.Intn(len(set.Blocks))
	block := set.Blocks[bIndex]
	if block.NextIndex != 0 {
		id = block.IDs[random.Intn(block.NextIndex)]
	}
	return
}

func main() {
	devices, err := cuda.AllDevices()
	if err != nil {
		// Handle error.
	}
	if len(devices) == 0 {
		// No devices found.
	}
	ctx, err = cuda.NewContext(devices[0], -1)
	if err != nil {
		fmt.Println("Fail to create context, due to", err)
		return
	}
	totalMem, err := devices[0].TotalMem()
	if BlockNum*BlockSize > totalMem {
		err = fs.ErrTooMuchGPUMemory
		return
	}

	cache, err := fs.NewCache(ctx, BlockNum, BlockSize)
	if err != nil {
		fmt.Println("Fail to init blocks, due to:", err)
		return
	}

	for i := 0; i < SetNum; i++ {
		set, err := fs.NewFeatureSet(cache, fmt.Sprintf("test%d", i), Dimension, Precision, Batch, 0, 0)
		if err != nil {
			fmt.Println("Fail to init feature set, due to:", err)
			return
		}

		r := rand.New(rand.NewSource(time.Now().Unix()))
		vec1 := make([]fs.Feature, 0)
		for i := 0; i < SetSize; i++ {
			var feature fs.Feature
			for j := 0; j < Dimension; j++ {
				feature.Value = append(feature.Value, r.Float32()*2-1)
			}
			feature.ID = fs.GetRandomString(12)
			vec1 = append(vec1, feature)
		}
		start := time.Now()
		sets = append(sets, set)
		err = set.Add(cache.BlockSize, vec1)
		if err != nil {
			fmt.Println("Fail to fill feature set, due to:", err)
			return
		}
		fmt.Printf("Init feature database for set (%d), use time %v \n", i, time.Since(start))
	}

	var wg sync.WaitGroup
	totalStart := time.Now()
	for s := 0; s < SetNum; s++ {
		wg.Add(1)
		go func(index int) {
			set := sets[index]
			defer wg.Done()
			handle, err := cublas.NewHandle(ctx)
			if err != nil {
				fmt.Println("Create handle error:", err)
				return
			}

			for b := 0; b < Round; b++ {
				start := time.Now()
				r := rand.New(rand.NewSource(time.Now().Unix()))
				var vec2 [][]float32
				for i := 0; i < Batch; i++ {
					var row []float32
					for j := 0; j < Dimension; j++ {
						row = append(row, r.Float32()*2-1)
					}
					vec2 = append(vec2, row)
				}
				_, err := set.Search(handle, vec2, 1)
				if err != nil {
					fmt.Println("Fail to search feature, err:", err)
				}
				fmt.Println("SetNum:", index, "Round ", b, " duration:", time.Since(start))
			}
		}(s)
	}

	aad new features
	wg.Add(1)
	go func() {
		for i := 0; i < 50; i++ {
			set := sets[i%SetNum]
			vec1 := make([]fs.Feature, 0)
			r := rand.New(rand.NewSource(time.Now().Unix()))
			size := r.Intn(10) + 1
			for i := 0; i < size; i++ {
				var feature fs.Feature
				for j := 0; j < Dimension; j++ {
					feature.Value = append(feature.Value, r.Float32()*2-1)
				}
				feature.ID = fs.GetRandomString(12)
				vec1 = append(vec1, feature)
			}
			start := time.Now()
			sets = append(sets, set)
			err = set.Add(cache.BlockSize, vec1)
			if err != nil {
				fmt.Println("Fail to fill feature set, due to:", err)
				return
			}
			fmt.Printf("Insert (%d) feature database for set (%d), use time %v \n", size, i%SetNum, time.Since(start))
			time.Sleep(time.Second)
		}
		wg.Done()
	}()

	random delete feature
	wg.Add(1)
	go func() {
		for i := 0; i < 200; i++ {
			set := sets[i%SetNum]
			id, _ := RandPickFeature(i % SetNum)
			if len(id) == 0 {
				id = fs.GetRandomString(12)
			}
			ids := []string{id}
			start := time.Now()
			deleted, err := set.Delete(ids)
			if err != nil {
				fmt.Println("Fail to delete feature ", id, ", err: ", err)
			}
			fmt.Printf("delete feature (%s) from set (%d),result: %v,  use time %v \n", id, i%SetNum, deleted, time.Since(start))
			time.Sleep(200 * time.Millisecond)
		}
		wg.Done()
	}()
	wg.Wait()
	fmt.Println("Total search time:", time.Since(totalStart))
}
