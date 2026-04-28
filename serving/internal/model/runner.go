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

	ort "github.com/yalue/onnxruntime_go"
)

type Runner struct {
	session      *ort.DynamicAdvancedSession
	featureOrder []string
	modelVersion string
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
	modelVersion := hex.EncodeToString(h[:])[:8]

	ort.SetSharedLibraryPath(sharedLibPath())
	if err := ort.InitializeEnvironment(); err != nil {
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

	return &Runner{
		session:      session,
		featureOrder: featureOrder,
		modelVersion: modelVersion,
	}, nil
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
	// probs[0] = P(class=0), probs[1] = P(class=1)
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
