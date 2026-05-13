package model

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type Runner struct {
	session      *ort.DynamicAdvancedSession
	featureOrder []string
	modelVersion string
}

// Sampler exposes the feature key set on one arbitrary Redis row so a model
// swap can fail fast if Redis is missing names a candidate model expects.
// nil disables the check.
type Sampler interface {
	SampleFeatureNames(ctx context.Context) ([]string, error)
}

// Swappable wraps a Runner so the underlying model can be replaced at runtime
// without restarting the process. Score holds an RLock and Reload takes the
// write lock, so a reload cannot proceed until in-flight Scores complete -
// which is the only point at which the old runner is safe to destroy.
type Swappable struct {
	mu      sync.RWMutex
	r       *Runner
	sampler Sampler
}

func NewSwappable(initial *Runner, sampler Sampler) *Swappable {
	return &Swappable{r: initial, sampler: sampler}
}

func (s *Swappable) Score(ctx context.Context, featureMap map[string]float32) (float32, string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.r.Score(ctx, featureMap)
}

func (s *Swappable) ModelVersion() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.r.modelVersion
}

func (s *Swappable) Reload(modelPath, featureOrderPath string) (string, error) {
	newR, err := NewRunner(modelPath, featureOrderPath)
	if err != nil {
		return "", err
	}
	if s.sampler != nil {
		if err := validateAgainstSample(s.sampler, newR.featureOrder); err != nil {
			newR.Destroy()
			return "", err
		}
	}
	s.mu.Lock()
	old := s.r
	s.r = newR
	s.mu.Unlock()
	old.Destroy()
	return newR.modelVersion, nil
}

func validateAgainstSample(sampler Sampler, featureOrder []string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	sampleNames, err := sampler.SampleFeatureNames(ctx)
	if err != nil {
		return fmt.Errorf("sample redis to validate new model: %w", err)
	}
	have := make(map[string]struct{}, len(sampleNames))
	for _, n := range sampleNames {
		have[n] = struct{}{}
	}
	var missing []string
	for _, name := range featureOrder {
		if _, ok := have[name]; !ok {
			missing = append(missing, name)
			if len(missing) >= 5 {
				break
			}
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("redis sample missing features required by new model (first %d shown): %v", len(missing), missing)
	}
	return nil
}

func (s *Swappable) Destroy() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.r != nil {
		s.r.Destroy()
		s.r = nil
	}
}

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func initORT() error {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(sharedLibPath())
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

func NewRunner(modelPath, featureOrderPath string) (*Runner, error) {
	featureOrderData, err := os.ReadFile(featureOrderPath)
	if err != nil {
		return nil, fmt.Errorf("read feature order: %w", err)
	}
	var featureOrder []string
	if err := json.Unmarshal(featureOrderData, &featureOrder); err != nil {
		return nil, fmt.Errorf("parse feature order: %w", err)
	}

	modelData, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("read model: %w", err)
	}
	h := sha256.Sum256(modelData)
	modelVersion := hex.EncodeToString(h[:])

	if err := initORT(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}

	inputNames := []string{"float_input"}
	outputNames := []string{"label", "probabilities"}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create onnx session: %w", err)
	}

	r := &Runner{
		session:      session,
		featureOrder: featureOrder,
		modelVersion: modelVersion,
	}

	// Catch feature_order.json / model mismatch at startup, not on the first live request.
	zeroMap := make(map[string]float32, len(featureOrder))
	for _, name := range featureOrder {
		zeroMap[name] = 0
	}
	if _, _, err := r.Score(context.Background(), zeroMap); err != nil {
		r.Destroy()
		return nil, fmt.Errorf("startup validation: model rejected %d-feature input from %s: %w", len(featureOrder), featureOrderPath, err)
	}

	return r, nil
}

func (r *Runner) Score(ctx context.Context, featureMap map[string]float32) (float32, string, error) {
	vec := make([]float32, len(r.featureOrder))
	for i, name := range r.featureOrder {
		v, ok := featureMap[name]
		if !ok {
			return 0, "", fmt.Errorf("missing feature %q in feature map", name)
		}
		vec[i] = v
	}

	hash := featureVecHash(vec)

	inputShape := ort.NewShape(1, int64(len(vec)))
	inputTensor, err := ort.NewTensor(inputShape, vec)
	if err != nil {
		return 0, "", fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	labelShape := ort.NewShape(1)
	labelTensor, err := ort.NewEmptyTensor[int64](labelShape)
	if err != nil {
		return 0, "", fmt.Errorf("create label tensor: %w", err)
	}
	defer labelTensor.Destroy()

	probShape := ort.NewShape(1, 2)
	probTensor, err := ort.NewEmptyTensor[float32](probShape)
	if err != nil {
		return 0, "", fmt.Errorf("create prob tensor: %w", err)
	}
	defer probTensor.Destroy()

	err = r.session.Run(
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{labelTensor, probTensor},
	)
	if err != nil {
		return 0, "", fmt.Errorf("onnx run: %w", err)
	}

	probs := probTensor.GetData()
	fraudProb := probs[1]
	return fraudProb, hash, nil
}

func (r *Runner) ModelVersion() string {
	return r.modelVersion
}

func (r *Runner) Destroy() {
	if r.session != nil {
		r.session.Destroy()
	}
}

func featureVecHash(vec []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, v := range vec {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(b[:], bits)
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func sharedLibPath() string {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}
	return "/usr/local/lib/libonnxruntime.so"
}
