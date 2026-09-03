// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package melspec

import "math"

// DBOptions configures PowerToDB.
type DBOptions struct {
	Ref   float64 // Reference power the result is relative to.
	AMin  float64 // Floor applied before taking the logarithm.
	TopDB float64 // Dynamic range; values more than TopDB below the peak are
	// clamped. Zero or negative disables clamping.
}

// DefaultDB returns librosa's power_to_db defaults: unit reference, a floor
// of 1e-10 and 80 dB of dynamic range.
//
// Note that the reference is 1, not the peak of the input. librosa's mfcc
// uses this default, so MFCCs are computed against an absolute reference
// rather than a per-signal maximum.
func DefaultDB() DBOptions {
	return DBOptions{Ref: 1, AMin: 1e-10, TopDB: 80}
}

// PowerToDB converts a power spectrogram to decibels in place, returning s.
func PowerToDB(s []float64, o DBOptions) []float64 {
	refDB := 10 * math.Log10(math.Max(o.AMin, math.Abs(o.Ref)))
	max := math.Inf(-1)
	for i, v := range s {
		d := 10*math.Log10(math.Max(o.AMin, v)) - refDB
		s[i] = d
		if d > max {
			max = d
		}
	}
	if o.TopDB > 0 {
		floor := max - o.TopDB
		for i, v := range s {
			if v < floor {
				s[i] = floor
			}
		}
	}
	return s
}

// dct2Ortho applies an orthonormal type-II DCT to src, writing the first
// len(dst) coefficients to dst. This matches scipy.fftpack.dct with
// type=2 and norm="ortho".
//
// The transform is evaluated directly in O(n*len(dst)); for the band counts
// used in speech features this is faster than an FFT-based route and avoids
// any dependence on transform length.
func dct2Ortho(dst, src []float64, cosTab []float64) {
	n := len(src)
	for k := range dst {
		var sum float64
		for i, v := range src {
			sum += v * cosTab[k*n+i]
		}
		if k == 0 {
			dst[k] = sum / math.Sqrt(float64(n))
		} else {
			dst[k] = sum * math.Sqrt(2/float64(n))
		}
	}
}

// cosTable precomputes cos(pi*k*(2i+1)/(2n)) for k < m, i < n.
func cosTable(m, n int) []float64 {
	t := make([]float64, m*n)
	for k := 0; k < m; k++ {
		for i := 0; i < n; i++ {
			t[k*n+i] = math.Cos(math.Pi * float64(k) * float64(2*i+1) / float64(2*n))
		}
	}
	return t
}

// MFCCOptions configures MFCC.
type MFCCOptions struct {
	N      int       // Number of cepstral coefficients to keep.
	Lifter float64   // Cepstral liftering coefficient; zero disables it.
	DB     DBOptions // Conversion applied to the mel spectrogram.
}

// DefaultMFCC returns librosa's mfcc defaults: 20 coefficients, no
// liftering, and a unit-reference decibel conversion.
func DefaultMFCC() MFCCOptions {
	return MFCCOptions{N: 20, Lifter: 0, DB: DefaultDB()}
}

// MFCC computes mel-frequency cepstral coefficients of x. The result has one
// row per frame and m.N columns.
func MFCC(x []float64, o Options, m MFCCOptions) Matrix {
	mel := MelSpectrogram(x, o)
	PowerToDB(mel.Data, m.DB)

	out := make([]float64, mel.Rows*m.N)
	tab := cosTable(m.N, mel.Cols)
	for t := 0; t < mel.Rows; t++ {
		dct2Ortho(out[t*m.N:(t+1)*m.N], mel.Row(t), tab)
	}

	if m.Lifter > 0 {
		// Emphasise higher cepstral coefficients, which are otherwise
		// small relative to the first few.
		for k := 0; k < m.N; k++ {
			g := 1 + (m.Lifter/2)*math.Sin(math.Pi*float64(k+1)/m.Lifter)
			for t := 0; t < mel.Rows; t++ {
				out[t*m.N+k] *= g
			}
		}
	}
	return Matrix{Data: out, Rows: mel.Rows, Cols: m.N}
}
