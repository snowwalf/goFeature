package goFeature

// FeatureValue : bytes in little endian
type FeatureValue []byte

// FeatureID : string id
type FeatureID string

// FeatureScore : match score, float32
type FeatureScore float32

// Feature : base struct for vector feture
type Feature struct {
	// feature value, dimesion * precison
	Value FeatureValue
	// unique index of feature
	ID FeatureID
}

// FeatureSearchResult : result for feature search
type FeatureSearchResult struct {
	// confidence for feature search
	Score FeatureScore
	// catched feature id
	ID FeatureID
}
