package goFeature

import (
	"math"
	"math/rand"
	"reflect"
	"sort"
	"time"
	"unsafe"
)

func FeatureToFeatureValue1D(features ...Feature) (ret FeatureValue) {
	for _, feature := range features {
		ret = append(ret, feature.Value...)
	}
	return
}

func FeatureValueTranspose1D(precision int, features ...FeatureValue) (ret FeatureValue, err error) {
	// little endian feature value
	if len(features) == 0 {
		return
	}
	if len(features[0])%precision != 0 {
		return nil, ErrBadTransposeValue
	}
	col := len(features[0]) / precision
	row := len(features)

	ret = make(FeatureValue, col*row*precision)
	for i := 0; i < col; i++ {
		for j := 0; j < row; j++ {
			copy(ret[(i*row+j)*precision:(i*row+j+1)*precision], features[j][i*precision:(i+1)*precision])
		}
	}
	return
}

func TFeatureValue(value interface{}) (FeatureValue, error) {

	switch value.(type) {
	case FeatureValue:
		return value.(FeatureValue), nil
	case []byte:
		return FeatureValue(value.([]byte)), nil
	case []int16:
		return *(*FeatureValue)(unsafe.Pointer(&reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&value.([]int16)[0])),
			Len:  len(value.([]int16)) * 2,
			Cap:  len(value.([]int16)) * 2,
		})), nil
	case []int32:
		return *(*FeatureValue)(unsafe.Pointer(&reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&value.([]int32)[0])),
			Len:  len(value.([]int32)) * 4,
			Cap:  len(value.([]int32)) * 4,
		})), nil
	case []float32:
		return *(*FeatureValue)(unsafe.Pointer(&reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&value.([]float32)[0])),
			Len:  len(value.([]float32)) * 4,
			Cap:  len(value.([]float32)) * 4,
		})), nil
	case []float64:
		return *(*FeatureValue)(unsafe.Pointer(&reflect.SliceHeader{
			Data: uintptr(unsafe.Pointer(&value.([]float64)[0])),
			Len:  len(value.([]float64)) * 8,
			Cap:  len(value.([]float64)) * 8,
		})), nil
	}
	return nil, ErrInvalidBufferData
}

func MaxNFloat32(vector []float32, limit int) ([]int, []float32) {
	type _result struct {
		Value float32
		Index int
	}
	var scores []_result
	for k, v := range vector {
		scores = append(scores, _result{Value: v, Index: k})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].Value > scores[j].Value })
	var index []int
	var max []float32
	for i := 0; i < limit && i < len(scores); i++ {
		index = append(index, scores[i].Index)
		max = append(max, scores[i].Value)
	}
	return index, max
}

func MaxNFeatureResult(vector []FeatureSearchResult, limit int) ([]int, []FeatureSearchResult) {
	type _result struct {
		Value FeatureSearchResult
		Index int
	}
	var scores []_result
	for k, v := range vector {
		scores = append(scores, _result{Value: v, Index: k})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].Value.Score > scores[j].Value.Score })
	var index []int
	var max []FeatureSearchResult
	for i := 0; i < limit && i < len(scores); i++ {
		index = append(index, scores[i].Index)
		max = append(max, scores[i].Value)
	}
	return index, max
}

func GetRandomString(length int) string {
	str := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-"
	bytes := []byte(str)
	result := []byte{}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < length; i++ {
		result = append(result, bytes[r.Intn(len(str))])
	}
	return string(result)
}

func NoramlizeFloat32(feature []float32) (ret []float32) {
	var mode float32
	for _, value := range feature {
		mode += value * value
	}
	mode = float32(math.Pow(float64(mode), 0.5))
	if mode == 0 {
		return make([]float32, len(feature), len(feature))
	}

	for _, value := range feature {
		ret = append(ret, value/mode)
	}
	return
}
