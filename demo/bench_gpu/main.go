package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"time"

	. "github.com/snowwalf/goFeature"
)

const (
	BLOCK_SIZE = 32 * 1024 * 1024 // 32M
	BATCH_SIZE = 100
)

var (
	mgr   Interface
	cache Cache
	set   string = "test_set"
)

func createFeatures(dim, pre, num int) (features []Feature, values []FeatureValue) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	for i := 0; i < num; i++ {
		var (
			feature Feature
			value   []float32
		)
		for j := 0; j < dim; j++ {
			value = append(value, r.Float32()*2-1)
		}
		feature.Value, _ = TFeatureValue(NoramlizeFloat32(value))
		feature.ID = FeatureID(GetRandomString(12))
		features = append(features, feature)
		values = append(values, feature.Value)
	}
	return
}

func createSet(ctx context.Context, dimension, precision, num, gpu_blocks int) (err error) {
	blockFeatureCount := BLOCK_SIZE / (dimension * precision)
	blockNum := (num + blockFeatureCount - 1) / blockFeatureCount
	if gpu_blocks == 0 {
		gpu_blocks = blockNum
	}
	if mgr, err = NewManager(ctx, 0, gpu_blocks*BLOCK_SIZE, blockNum, BLOCK_SIZE); err != nil {
		fmt.Println("Fail to init feature set, due to:", err)
		return err
	}

	fmt.Printf("block size: %v , block num: %v \n", blockFeatureCount, blockNum)

	if err = mgr.NewSet(set, dimension, precision, BATCH_SIZE); err != nil {
		fmt.Println("Fail to init feature set, due to:", err)
		return err
	}

	features, _ := createFeatures(dimension, precision, blockFeatureCount)
	for i := 0; i < num/blockFeatureCount; i++ {
		if err = mgr.AddFeature(set, features...); err != nil {
			fmt.Println("Fail to fill feature set, due to:", err)
			return err
		}
	}
	if remain := num % blockFeatureCount; remain > 0 {
		if err = mgr.AddFeature(set, features[:remain]...); err != nil {
			fmt.Println("Fail to fill feature set, due to:", err)
			return err
		}
	}
	return
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	ctx := context.Background()
	total := flag.Int("total", 10000, "total num of features")
	num := flag.Int("num", 1, "search num")
	dimension := flag.Int("dimension", 512, "dimension of feature, 512 or 4096")
	times := flag.Int("times", 50, "search times")
	verbose := flag.Bool("verbose", false, "verbose")
	block := flag.Int("block", 0, "gpu block number")
	flag.Parse()

	precision := 4

	if err := createSet(ctx, *dimension, precision, *total, *block); err != nil {
		fmt.Println("create set failed due to ", err)
		return
	}

	fmt.Printf("searching %v in %v \n", *num, *total)

	_, values := createFeatures(*dimension, precision, *num)

	start := time.Now()
	for i := 0; i < *times; i++ {
		start := time.Now()
		ret, err := mgr.Search(set, -1.0, 1, values...)
		if err != nil {
			fmt.Println("search failed due to ", err)
			return
		}
		if *verbose {
			fmt.Printf("%v, %.6f\n", i, float64((time.Now().UnixNano()-start.UnixNano()))/1e9)
			fmt.Println(ret)
		}
	}
	totalTimeCost := float64((time.Now().UnixNano() - start.UnixNano())) / 1e9
	fmt.Printf("total: %.6f, average %.6f\n", totalTimeCost, totalTimeCost/(float64(*times)))
}
