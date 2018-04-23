package goFeature

// Interface : user interface of goFeature manager
type Interface interface {

	// ----------------------------------------------------------------------------------- //
	// Interfaces of Set                                                                   //
	// ----------------------------------------------------------------------------------- //

	// NewSet: create set
	//  - name: set name, unique
	//  - dims: dimension of feature
	//  - precision: precision of feature
	//  - batch: max batch size of feature search
	NewSet(name string, dims int, precision int, batch int) error

	// DestroySet: destroy the set, release all the resource accquired
	//	- name: set name, unique
	DestroySet(name string) error

	// GetSet: read the set info
	//  - name: set name, unique
	GetSet(name string) (dimesion, precision, batch int, err error)

	// UpdateSet: update set config, only batch is available
	//  - name: set name, unique
	//  - batch: new batch size
	UpdateSet(name string, batch int) error

	// ----------------------------------------------------------------------------------- //
	// Interfaces of Feature                                                               //
	// ----------------------------------------------------------------------------------- //

	// AddFeature: add features to set
	//  - name: set name, unique
	// 	- features: features to be added
	AddFeature(name string, features ...Feature) error

	// DeleteFeature: try to delete features by id
	//  - name: set name, unique
	//  - ids: feature id to be deleted
	//  - deleted: feature ids deleted after function calling, nil if no one found
	DeleteFeature(name string, ids ...FeatureID) (deleted []FeatureID, err error)

	// UpdateFeature: try to update features
	//  - name: set name, unique
	//  - features: feature to be deleted
	//  - updated: feature ids updated after function calling, nil if no one found
	UpdateFeature(name string, features ...Feature) (updated []FeatureID, err error)

	// ReadFeautre: try to read features by id
	//  - name: set name, unique
	//  - ids: feature id to be read
	//  - features: feature got after function calling, nil if no one found
	ReadFeautre(name string, ids ...FeatureID) (features []Feature, err error)

	// Search: search targe features
	//  - name: set name, unique
	//  - threshold: score threshold for search
	//	- limit: top N result
	//	- features: target features value
	//	- ret: search results
	Search(name string, threshold FeatureScore, limit int, features ...FeatureValue) (ret [][]FeatureSearchResult, err error)
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
