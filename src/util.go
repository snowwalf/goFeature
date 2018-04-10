// +build cublas

package goFeature

import (
	"math/rand"
	"sort"
	"time"
)

func Float32Array2DTo1D(array [][]float32) (ret []float32) {
	for _, row := range array {
		ret = append(ret, row...)
	}
	return
}

func Float32FeautureTO1D(features []Feature) (ret []float32) {
	for _, feature := range features {
		ret = append(ret, feature.Value...)
	}
	return
}

// 二维特征矩阵 转置成 一维列优先矩阵
func TFloat32(vector [][]float32, col, row int) []float32 {
	ret := make([]float32, col*row)
	for i := 0; i < col; i++ {
		for j := 0; j < row; j++ {
			ret[i*row+j] = vector[j][i]
		}
	}
	return ret
}

func MaxFloat32(vector []float32) (int, float32) {
	var max float32
	max = -99999.9
	var index int
	index = -1
	for i, value := range vector {
		if value > max {
			index = i
			max = value
		}
	}
	return index, max
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

func MaxNFeatureResult(vector []FeatureResult, limit int) ([]int, []FeatureResult) {
	type _result struct {
		Value FeatureResult
		Index int
	}
	var scores []_result
	for k, v := range vector {
		scores = append(scores, _result{Value: v, Index: k})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].Value.Score > scores[j].Value.Score })
	var index []int
	var max []FeatureResult
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
