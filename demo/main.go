// +build cublas

package main

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/snowwalf/goFeature"
	"github.com/unixpickle/cuda"
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
	ctx      *cuda.Context
	sets     []goFeature.Set
	features [][]goFeature.FeatureID
)

func RandPickFeature(index int) (id goFeature.FeatureID, err error) {
	if len(sets) < index {
		err = errors.New("RandPickFeature: index is out of bound")
		return
	}
	ids := features[index]
	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	id = ids[random.Intn(len(ids))]
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
		err = goFeature.ErrTooMuchGPUMemory
		return
	}
	var (
		cache goFeature.Cache
	)

	cache, err = goFeature.NewCache(ctx, BlockNum, BlockSize)
	if err != nil {
		fmt.Println("Fail to init blocks, due to:", err)
		return
	}

	for i := 0; i < SetNum; i++ {
		var ids []goFeature.FeatureID
		name := fmt.Sprintf("test%d", i)
		err := cache.NewSet(name, Dimension, Precision, Batch)
		if err != nil {
			fmt.Println("Fail to init feature set, due to:", err)
			return
		}

		r := rand.New(rand.NewSource(time.Now().Unix()))
		vec1 := make([]goFeature.Feature, 0)
		for i := 0; i < SetSize; i++ {
			var (
				values  []float32
				feature goFeature.Feature
			)
			for j := 0; j < Dimension; j++ {
				values = append(values, r.Float32()*2-1)
			}
			feature.Value, _ = goFeature.TFeatureValue(values)
			feature.ID = goFeature.FeatureID(goFeature.GetRandomString(12))
			vec1 = append(vec1, feature)
			ids = append(ids, feature.ID)
		}
		start := time.Now()
		set, err := cache.GetSet(name)
		if err != nil {
			fmt.Println("Fail to get feature set", name, ", error:", err)
			return
		}
		err = set.Add(vec1...)
		if err != nil {
			fmt.Println("Fail to fill feature set, due to:", err)
			return
		}
		sets = append(sets, set)
		features = append(features, ids)
		fmt.Printf("Init feature database for set (%d), use time %v \n", i, time.Since(start))
	}

	var wg sync.WaitGroup
	totalStart := time.Now()
	for s := 0; s < SetNum; s++ {
		wg.Add(1)
		go func(index int) {
			set := sets[index]
			defer wg.Done()

			for b := 0; b < Round; b++ {
				start := time.Now()
				r := rand.New(rand.NewSource(time.Now().Unix()))
				var vec2 []goFeature.FeatureValue
				for i := 0; i < Batch; i++ {
					var row []float32
					for j := 0; j < Dimension; j++ {
						row = append(row, r.Float32()*2-1)
					}
					value, _ := goFeature.TFeatureValue(row)
					vec2 = append(vec2, value)
				}
				_, err := set.Search(0, 1, vec2...)
				if err != nil {
					fmt.Println("Fail to search feature, err:", err)
				}
				fmt.Println("SetNum:", index, "Round ", b, " duration:", time.Since(start))
			}
		}(s)
	}

	// aad new features
	wg.Add(1)
	go func() {
		for i := 0; i < 50; i++ {
			var ids []goFeature.FeatureID
			set := sets[i%SetNum]
			vec1 := make([]goFeature.Feature, 0)
			r := rand.New(rand.NewSource(time.Now().Unix()))
			size := r.Intn(10) + 1
			for i := 0; i < size; i++ {
				var (
					value   []float32
					feature goFeature.Feature
				)
				for j := 0; j < Dimension; j++ {
					value = append(value, r.Float32()*2-1)
				}
				feature.Value, _ = goFeature.TFeatureValue(value)
				feature.ID = goFeature.FeatureID(goFeature.GetRandomString(12))
				vec1 = append(vec1, feature)
				ids = append(ids, feature.ID)
			}
			start := time.Now()
			sets = append(sets, set)
			err = set.Add(vec1...)
			if err != nil {
				fmt.Println("Fail to fill feature set, due to:", err)
				return
			}
			features[i%SetNum] = append(features[i%SetNum], ids...)
			fmt.Printf("Insert (%d) feature database for set (%d), use time %v \n", size, i%SetNum, time.Since(start))
			time.Sleep(time.Second)
		}
		wg.Done()
	}()

	// random delete feature
	wg.Add(1)
	go func() {
		for i := 0; i < 200; i++ {
			set := sets[i%SetNum]
			id, _ := RandPickFeature(i % SetNum)
			if len(id) == 0 {
				id = goFeature.FeatureID(goFeature.GetRandomString(12))
			}
			ids := []goFeature.FeatureID{id}
			start := time.Now()
			deleted, err := set.Delete(ids...)
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
