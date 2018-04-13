// +build cublas

package goFeature

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/unixpickle/cuda"
	"github.com/unixpickle/cuda/cublas"
)

const (
	blockSize = 1000 * 512 * 4
	blockNum  = 10
	dims      = 512
	premision = 4
	count     = 1000
	batch     = 100
)

var (
	ctx    *cuda.Context
	cache  Cache
	set    Set
	handle *cublas.Handle
)

func init() {
	var err error
	devices, err := cuda.AllDevices()
	if err != nil {
		// Handle error.
	}
	if len(devices) == 0 {
		// No devices found.
	}
	ctx, err = cuda.NewContext(devices[0], -1)
	if err != nil || ctx == nil {
		panic("init cuda failed")
	}

	cache, err = NewCache(ctx, blockNum, blockSize)
	if err != nil {
		panic(fmt.Sprint("Fail to init blocks, due to:", err))
	}

	name := "search_benchmark"
	err = cache.NewSet(name, dims, premision, batch)
	if err != nil {
		panic(fmt.Sprint("Fail to init feature set, due to:", err))
	}
	set, _ = cache.GetSet(name)

	r := rand.New(rand.NewSource(time.Now().Unix()))
	vec1 := make([]Feature, 0)
	for i := 0; i < count; i++ {
		var (
			feature Feature
			value   []float32
		)
		for j := 0; j < dims; j++ {
			value = append(value, r.Float32()*2-1)
		}
		feature.Value, _ = TByte(value)
		feature.ID = FeatureID(GetRandomString(12))
		vec1 = append(vec1, feature)
	}
	if err = set.Add(vec1...); err != nil {
		panic(fmt.Sprint("Fail to fill feature set, due to:", err))
	}

}

// search 1 in 1000
func BenchmarkSearch1TO1000(b *testing.B) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	var target FeatureValue
	var row []float32
	for j := 0; j < dims; j++ {
		row = append(row, r.Float32()*2-1)
	}
	target, _ = TByte(row)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := set.Search(0.0, 1, target); err != nil {
			b.Fatalf("failed to search, err: ", err)
		}
	}
}
