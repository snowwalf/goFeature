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
	blockSize = 10000 * 512 * 4
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
		feature.Value, _ = TFeatureValue(value)
		feature.ID = FeatureID(GetRandomString(12))
		vec1 = append(vec1, feature)
	}
	if err = set.Add(vec1...); err != nil {
		panic(fmt.Sprint("Fail to fill feature set, due to:", err))
	}

}

func TestBasicSearch(t *testing.T) {
	var (
		err error
		ret [][]FeatureSearchResult
	)
	if err = cache.NewSet("basic_search", 5, 4, 5); err != nil {
		panic(fmt.Sprint("Fail to init feature set, due to:", err))
	}
	set, _ := cache.GetSet("basic_search")

	v1 := NoramlizeFloat32([]float32{1.0, 2.0, 3.0, 4.0, 5.0})
	v2 := NoramlizeFloat32([]float32{2.0, 1.0, -3.0, 2.1, -1.0})

	f1 := Feature{
		ID: FeatureID(GetRandomString(12)),
	}
	f1.Value, _ = TFeatureValue(v1)
	f2 := Feature{
		ID: FeatureID(GetRandomString(12)),
	}
	f2.Value, _ = TFeatureValue(v2)
	if err = set.Add(f1, f2); err != nil {
		panic(fmt.Sprint("Fail to fill feature set, due to:", err))
	}
	// search f1
	if ret, err = set.Search(0, 1, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 1 || ret[0][0].ID != f1.ID || ret[0][0].Score < 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
	}

	v3 := make([]FeatureValue, 0)
	v3 = append(v3, f1.Value)
	v3 = append(v3, f2.Value)
	if ret, err = set.Search(-1, 5, v3...); err != nil {
		panic(fmt.Sprint("Fail to fill search multi-features, due to:", err))
	}
	if len(ret) != 2 || len(ret[0]) != 2 || len(ret[1]) != 2 || ret[0][0].Score < 0.999999 || ret[0][0].ID != f1.ID || ret[1][0].Score < 0.999999 || ret[1][0].ID != f2.ID {
		panic(fmt.Sprint("Fail to fill search multi-features got wrong targed, ret:", ret))
	}

	// search f1, limit=2
	if ret, err = set.Search(-1, 2, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 2 || ret[0][0].ID != f1.ID || ret[0][0].Score < 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
	}

	// search f1, limit=2, threshold = 0.99
	if ret, err = set.Search(0.99, 2, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 1 || ret[0][0].ID != f1.ID || ret[0][0].Score < 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
	}

	deleted, err := set.Delete(f1.ID)
	if err != nil || len(deleted) != 1 || deleted[0] != f1.ID {
		panic(fmt.Sprint("Fail to delete f1, error:", err))
	}
	if ret, err = set.Search(-1, 2, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 1 || ret[0][0].ID != f2.ID || ret[0][0].Score > 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
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
	target, _ = TFeatureValue(row)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := set.Search(0.0, 1, target); err != nil {
			b.Fatalf("failed to search, err: ", err)
		}
	}
}
