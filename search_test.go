// +build cublas

package goFeature

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

const (
	blockSize   = 500 * 512 * 4
	blockNum    = 10
	gpuBlockNum = 5
	dims        = 512
	premision   = 4
	count       = 1000
	batch       = 5
)

var (
	mgr   Interface
	cache Cache
	set   Set
)

func init() {
	var (
		err error
		ctx = context.Background()
	)
	mgr, err = NewManager(ctx, 0, blockSize*gpuBlockNum, blockNum, blockSize)
	if err != nil {
		panic("fail to create manager")
	}
}

func TestBasicSearch(t *testing.T) {
	var (
		err      error
		ret      [][]FeatureSearchResult
		set      = "basic_search"
		features []Feature
	)
	if err = mgr.NewSet(set, dims, premision, batch); err != nil {
		panic(fmt.Sprint("Fail to init feature set, due to:", err))
	}

	r := rand.New(rand.NewSource(time.Now().Unix()))
	for i := 0; i < count; i++ {
		var (
			feature Feature
			value   []float32
		)
		for j := 0; j < dims; j++ {
			value = append(value, r.Float32()*2-1)
		}
		feature.Value, _ = TFeatureValue(NoramlizeFloat32(value))
		feature.ID = FeatureID(GetRandomString(12))
		features = append(features, feature)
	}

	if err = mgr.AddFeature(set, features...); err != nil {
		panic(fmt.Sprint("Fail to fill feature set, due to:", err))
	}
	// search f1
	f1 := features[0]
	if ret, err = mgr.Search(set, -1, 1, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 1 || ret[0][0].ID != f1.ID || ret[0][0].Score < 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
	}

	// search f1, limit=2
	if ret, err = mgr.Search(set, -1, 2, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 2 || ret[0][0].ID != f1.ID || ret[0][0].Score < 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
	}

	// search f1, limit=2, threshold = 0.99
	if ret, err = mgr.Search(set, 0.99, 2, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 1 || ret[0][0].ID != f1.ID || ret[0][0].Score < 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
	}

	// search f1,f2
	f2 := features[1]
	if ret, err = mgr.Search(set, -1, 5, f1.Value, f2.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search multi-features, due to:", err))
	}
	if len(ret) != 2 || len(ret[0]) != 5 || len(ret[1]) != 5 || ret[0][0].Score < 0.999999 || ret[0][0].ID != f1.ID || ret[1][0].Score < 0.999999 || ret[1][0].ID != f2.ID {
		panic(fmt.Sprint("Fail to fill search multi-features got wrong targed, ret:", ret))
	}

	deleted, err := mgr.DeleteFeature(set, f1.ID)
	if err != nil || len(deleted) != 1 || deleted[0] != f1.ID {
		panic(fmt.Sprint("Fail to delete f1, error:", err))
	}
	if ret, err = mgr.Search(set, -1, 2, f1.Value); err != nil {
		panic(fmt.Sprint("Fail to fill search feature, due to:", err))
	}
	if len(ret) != 1 || len(ret[0]) != 2 || ret[0][0].ID == f1.ID || ret[0][0].Score > 0.999999 {
		panic(fmt.Sprint("Fail to search feature got wrong target, ret:", ret))
	}
}
