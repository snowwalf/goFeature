// +build cublas

package goFeature

import "github.com/unixpickle/cuda/cublas"

// Cache : interface of cache, the main object of features
type Cache interface {
	NewSet(string, int, int, int) error
	DestroySet(string) error
	GetSet(string) (Set, error)
	GetBlockSize() int
	GetEmptyBlock(int) ([]Block, error)
}

// Set : interface of set
type Set interface {
	Add(...Feature) error
	Delete(...FeatureID) ([]FeatureID, error)
	Update(...Feature) ([]FeatureID, error)
	Read(...FeatureID) ([]Feature, error)
	Destroy() error

	Search(float32, int, ...FeatureValue) ([][]FeatureSearchResult, error)
}

// Block : interface of block, the basic scheuling unit
type Block interface {
	Capacity() int
	Margin() int
	IsOwned() bool
	Accquire(*cublas.Handle, string, int, int) error
	Release() error
	Insert(...Feature) error
	Delete(...FeatureID) ([]FeatureID, error)
	Update(...FeatureID) ([]FeatureID, error)
	Read(...FeatureID) ([]Feature, error)

	Search(Buffer, Buffer, int, int) ([][]FeatureSearchResult, error)
}

// Buffer : buffer in memory for both CPU and GPU
type Buffer interface {
	GetBuffer() interface{}
	Write(FeatureValue) error
	Read() (FeatureValue, error)
	Copy(Buffer) error
	Reset() error
	Slice(int, int) (Buffer, error)
	Size() int
}
