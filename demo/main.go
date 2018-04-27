package main

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/snowwalf/goFeature"
)

const (
	//BlockSize : size of feature block
	BlockSize      = 1024 * 1024 * 32
	BlockNum       = 100
	GPUBlockNum    = 50
	Dimension      = 512
	Batch          = 5
	Precision      = 4
	Round          = 500
	SetSize        = 500000
	SetNum         = 1
	SearchParallel = 1
	InitParallel   = 10
)

var (
	//ids      [][]goFeature.FeatureID
	sets     []string
	features [][]goFeature.FeatureID
	delay    int64
)

func RandPickFeature(index int) (id goFeature.FeatureID, err error) {
	ids := features[index]
	random := rand.New(rand.NewSource(time.Now().UnixNano()))
	id = ids[random.Intn(len(ids))]
	return
}

func main() {
	ctx := context.Background()
	mgr, err := goFeature.NewManager(ctx, 2, BlockSize*GPUBlockNum, BlockNum, BlockSize)
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
				r := rand.New(rand.NewSource(time.Now().Unix()))
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

	totalStart := time.Now()
	for s := 0; s < SetNum; s++ {
		wg.Add(1)
		for p := 0; p < SearchParallel; p++ {
			go func(index, parallel int) {
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
					_, err := mgr.Search(set, 0, 1, vec2...)
					if err != nil {
						fmt.Println("Fail to search feature, err:", err)
					}
					fmt.Println("SetNum:", index, "parallel:", parallel, "Round ", b, " duration:", time.Since(start))
					atomic.AddInt64(&delay, int64(time.Since(start)))
				}
			}(s, p)
		}
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
	fmt.Println("Average search delay:", time.Duration(delay/(SearchParallel*SetNum*Round)))
	fmt.Println("Total search time:", time.Since(totalStart))
	fmt.Println("QPS:", float64(Round*SearchParallel*SetNum*Batch)/(time.Since(totalStart).Seconds()))
}
