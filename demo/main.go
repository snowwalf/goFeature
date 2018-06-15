package main

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/snowwalf/goFeature"
)

const (
	//BlockSize : size of feature block
	BlockSize    = 1024 * 1024 * 32
	BlockNum     = 200
	GPUBlockNum  = 40
	Dimension    = 512
	Precision    = 4
	Round        = 100
	SetSize      = 16 * 1024 * 200
	SetNum       = 1
	InitParallel = 10
	Batch        = 1
)

var (
	SearchParallel = 1
	sets           []string
	features       [][]goFeature.FeatureID
	delay          int64
)

func RandPickFeature(index int) (id goFeature.FeatureID, err error) {
	ids := features[index]
	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	id = ids[random.Intn(len(ids))]
	return
}

func main() {
	ctx := context.Background()
	mgr, err := goFeature.NewManager(ctx, 0, BlockSize*GPUBlockNum, BlockNum, BlockSize)
	if err != nil {
		fmt.Println("Fail to create manager, due to:", err)
		return
	}

	var (
		wg  sync.WaitGroup
		mut sync.Mutex
	)
	for i := 0; i < SetNum; i++ {
		var ids []goFeature.FeatureID
		name := fmt.Sprintf("test%d", i)
		err := mgr.NewSet(name, Dimension, Precision, Batch)
		if err != nil {
			fmt.Println("Fail to init feature set, due to:", err)
			return
		}
		start := time.Now()

		for k := 0; k < InitParallel; k++ {
			wg.Add(1)
			go func(size int) {
				r := rand.New(rand.NewSource(time.Now().UnixNano()))
				var (
					values  []float32
					feature goFeature.Feature
				)
				for j := 0; j < Dimension; j++ {
					values = append(values, r.Float32()*2-1)
				}
				feature.Value, _ = goFeature.TFeatureValue(values)
				feature.ID = goFeature.FeatureID(goFeature.GetRandomString(12))
				vec1 := make([]goFeature.Feature, 0)
				for i := 0; i < 1024; i++ {
					vec1 = append(vec1, feature)
					ids = append(ids, feature.ID)
				}
				for size > 1024 {
					err = mgr.AddFeature(name, vec1...)
					if err != nil {
						fmt.Println("Fail to fill feature set, due to:", err)
						return
					}

					mut.Lock()
					sets = append(sets, name)
					features = append(features, ids)
					mut.Unlock()
					size -= 1024
				}
				if size > 0 {
					err = mgr.AddFeature(name, vec1[:size]...)
					if err != nil {
						fmt.Println("Fail to fill feature set, due to:", err)
						return
					}

					mut.Lock()
					sets = append(sets, name)
					features = append(features, ids)
					size -= 1024
					mut.Unlock()
				}
				wg.Done()
			}(SetSize / InitParallel)
		}

		wg.Wait()
		fmt.Printf("Init feature database for set (%d), use time %v \n", i, time.Since(start))
	}
	runtime.GC()
	for t := 0; t < 10; t++ {
		SearchParallel = t + 1
		totalStart := time.Now()
		delay = 0
		for s := 0; s < SetNum; s++ {
			for p := 0; p < SearchParallel; p++ {
				wg.Add(1)
				go func(index, parallel int) {
					set := sets[index]
					defer wg.Done()
					for b := 0; b < Round; b++ {
						start := time.Now()
						r := rand.New(rand.NewSource(time.Now().UnixNano()))
						var vec2 []goFeature.FeatureValue
						for i := 0; i < Batch; i++ {
							var row []float32
							for j := 0; j < Dimension; j++ {
								row = append(row, r.Float32()*2-1)
							}
							value, _ := goFeature.TFeatureValue(row)
							vec2 = append(vec2, value)
						}
						_, err := mgr.Search(set, -1, 1, vec2...)
						if err != nil {
							fmt.Println("Fail to search feature, err:", err)
						}
						//fmt.Println("t:", t, "SetNum:", index, "parallel:", parallel, "Round ", b, " duration:", time.Since(start))
						atomic.AddInt64(&delay, int64(time.Since(start)))
					}
				}(s, p)
			}
			wg.Wait()
		}
		fmt.Println("SearchParallel:", SearchParallel,
			"delay:", time.Duration(delay),
			"delay/s:", time.Duration(delay/int64(SearchParallel*SetNum*Round)),
			"QPS:", float64(Round*SearchParallel*SetNum*Batch)/(time.Since(totalStart).Seconds()),
			"Total search time:", time.Since(totalStart))
	}

	//aad new features
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
			err = mgr.AddFeature(set, vec1...)
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
			deleted, err := mgr.DeleteFeature(set, ids...)
			if err != nil {
				fmt.Println("Fail to delete feature ", id, ", err: ", err)
			}
			fmt.Printf("delete feature (%s) from set (%d),result: %v,  use time %v \n", id, i%SetNum, deleted, time.Since(start))
			time.Sleep(200 * time.Millisecond)
		}
		wg.Done()
	}()
	wg.Wait()
}
