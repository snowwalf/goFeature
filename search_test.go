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
	cache  *Cache
	set    *FeatureSet
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

	handle, err = cublas.NewHandle(ctx)
	if err != nil {
		panic(fmt.Sprint("Create handle error:", err))
	}

	cache, err = NewCache(ctx, blockNum, blockSize)
	if err != nil {
		panic(fmt.Sprint("Fail to init blocks, due to:", err))
	}

	set, err = NewFeatureSet(cache, "search_benchmark", dims, premision, batch, 0, 0)
	if err != nil {
		panic(fmt.Sprint("Fail to init feature set, due to:", err))
	}

	r := rand.New(rand.NewSource(time.Now().Unix()))
	vec1 := make([]Feature, 0)
	for i := 0; i < count; i++ {
		var feature Feature
		for j := 0; j < dims; j++ {
			feature.Value = append(feature.Value, r.Float32()*2-1)
		}
		feature.ID = GetRandomString(12)
		vec1 = append(vec1, feature)
	}
	if err = set.Add(cache.BlockSize, vec1); err != nil {
		panic(fmt.Sprint("Fail to fill feature set, due to:", err))
	}

	r = rand.New(rand.NewSource(time.Now().Unix()))

}

// search 1 in 1000
func BenchmarkSearch1TO1000(b *testing.B) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	var target [][]float32
	var row []float32
	for j := 0; j < dims; j++ {
		row = append(row, r.Float32()*2-1)
	}
	target = append(target, row)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := set.Search(handle, target, 1); err != nil {
			b.Fatalf("failed to search, err: ", err)
		}
	}
}

func BenchmarkSearch10TO1000(b *testing.B) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	var target [][]float32
	var row []float32
	for i := 0; i < 10; i++ {
		for j := 0; j < dims; j++ {
			row = append(row, r.Float32()*2-1)
		}
		target = append(target, row)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := set.Search(handle, target, 1); err != nil {
			b.Fatalf("failed to search, err: ", err)
		}
	}
}

func BenchmarkSearch20TO1000(b *testing.B) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	var target [][]float32
	var row []float32
	for i := 0; i < 20; i++ {
		for j := 0; j < dims; j++ {
			row = append(row, r.Float32()*2-1)
		}
		target = append(target, row)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := set.Search(handle, target, 1); err != nil {
			b.Fatalf("failed to search, err: ", err)
		}
	}
}

func BenchmarkSearch50TO1000(b *testing.B) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	var target [][]float32
	var row []float32
	for i := 0; i < 50; i++ {
		for j := 0; j < dims; j++ {
			row = append(row, r.Float32()*2-1)
		}
		target = append(target, row)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := set.Search(handle, target, 1); err != nil {
			b.Fatalf("failed to search, err: ", err)
		}
	}
}

func BenchmarkSearch100TO1000(b *testing.B) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	var target [][]float32
	var row []float32
	for i := 0; i < 100; i++ {
		for j := 0; j < dims; j++ {
			row = append(row, r.Float32()*2-1)
		}
		target = append(target, row)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := set.Search(handle, target, 1); err != nil {
			b.Fatalf("failed to search, err: ", err)
		}
	}
}
