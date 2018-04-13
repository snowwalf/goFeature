// +build cublas

package goFeature

import "github.com/unixpickle/cuda/cublas"

// Cache : interface of cache, the main object of features
type Cache interface {
	// NewSet: create set
	//  - name: set name, unique
	//  - dims: dimension of feature
	//  - precision: precision of feature
	//  - batch: max batch size of feature search
	NewSet(name string, dims int, precision int, batch int) error

	// DestroySet: destroy the set, release all the resource accquired
	//	- name: set name, unique
	DestroySet(name string) error

	// GetSet: get set by name
	//	- name: set name, unique
	GetSet(name string) (Set, error)

	// GetBlockSize: get the blocksize of linked blocks
	GetBlockSize() int

	// GetEmptyBlock: try to get blocks which not accquired
	//  - blocknum: block number to be accquired
	GetEmptyBlock(blocknum int) ([]Block, error)
}

// Set : interface of set
type Set interface {
	// Add: add features to set
	// 	- features: features to be added
	Add(features ...Feature) error

	// Delete: try to delete features by id
	//  - ids: feature id to be deleted
	//  - deleted: feature ids deleted after function calling, nil if no one found
	Delete(ids ...FeatureID) (deleted []FeatureID, err error)

	// Update: try to update features
	//  - features: feature to be deleted
	//  - updated: feature ids updated after function calling, nil if no one found
	Update(features ...Feature) (updated []FeatureID, err error)

	// Read: try to read features by id
	//  - ids: feature id to be read
	//  - features: feature got after function calling, nil if no one found
	Read(ids ...FeatureID) (features []Feature, err error)

	// Destroy: destroy the set, release all the blocks accquired
	Destroy() error

	// Search: search targe features
	//  - threshold: score threshold for search
	//	- limit: top N result
	//	- features: target features value
	//	- ret: search results
	Search(threshold FeatureScore, limit int, features ...FeatureValue) (ret [][]FeatureSearchResult, err error)
}

// Block : interface of block, the basic scheuling unit
type Block interface {
	// Capacity: get max feautre number of the block
	Capacity() int

	// Margin: number of features can be inserted
	Margin() int

	// IsOwned: check if accquired
	IsOwned() bool

	// Accquire: one set tries to accquire the block
	//  - handle: cublas handle created by set
	//  - owner: set name, unique
	//  - dims: dimension of feature
	//  - precision: precision of feature
	Accquire(handle *cublas.Handle, owner string, dims int, premision int) error

	// Release: release the accquired block
	Release() error

	// Insert: insert features into the block
	//  - features: features to be inserted
	Insert(features ...Feature) error

	// Delete: try to delete features by id
	//  - ids: feature id to be deleted
	//  - deleted: feature ids deleted after function calling, nil if no one found
	Delete(...FeatureID) (deleted []FeatureID, err error)

	// Update: try to update features
	//  - features: feature to be deleted
	//  - updated: feature ids updated after function calling, nil if no one found
	Update(features ...Feature) (updated []FeatureID, err error)

	// Read: try to read features by id
	//  - ids: feature id to be read
	//  - features: feature got after function calling, nil if no one found
	Read(ids ...FeatureID) (features []Feature, err error)

	// Search: search targe features
	//  - inputBuffer: buffer stored target features value
	//  - outputBuffer: buffer to store search result, temporary
	//  - batch: max search batch size
	//	- limit: top N result
	//	- ret: search results
	Search(inputBuffer, outputBuffer Buffer, batch int, limit int) (ret [][]FeatureSearchResult, err error)
}

// Buffer : buffer in memory for both CPU and GPU
type Buffer interface {
	// GetBuffer: get the real buffer interface
	GetBuffer() interface{}

	// Write: write feature value into buffer
	Write(value FeatureValue) error

	// Read: read the feature value stored
	Read() (value FeatureValue, err error)

	// Copy: copy the src into the buffer
	Copy(src Buffer) error

	// Reset: clear the buffer
	Reset() error

	// Slice: try to slice the buffer, [start:end)
	//  - start: start index
	//  - end: end index
	Slice(start, end int) (Buffer, error)

	// Size: get the size of the buffer
	Size() int
}
