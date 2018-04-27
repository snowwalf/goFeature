// +build !cublas

package goFeature

import "context"

type Core struct {
}

func initCuda(gpuID, gpuMemSize int) (Buffer, error) {
	panic("Should not call initCuda on cpu")
	return nil, nil
}

func NewCore(buffer Buffer) (*Core, error) {
	panic("Should not call NewCore on cpu")
	return nil, nil
}

func (c *Core) Work(ctx context.Context) {
	panic("Should not call Work on cpu")
	return
}

func (c *Core) Do(SearchJob) {
	panic("Should not call Do on cpu")
}

func loadFeatures(features ...FeatureValue) (Buffer, error) {
	panic("Should not call loadFeatures on cpu")
	return nil, nil
}
