// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package melspec

import "math"

// Scale selects the frequency-to-mel warping.
type Scale int

const (
	// Slaney is the mel scale from Slaney's Auditory Toolbox: linear below
	// 1 kHz and logarithmic above. It corresponds to librosa's htk=False.
	Slaney Scale = iota

	// HTK is the mel scale used by HTK and described in the HTK Book,
	//
	//	m = 2595*log10(1 + f/700)
	//
	// It corresponds to librosa's htk=True.
	HTK
)

// Constants of the Slaney scale. Below breakFreq the mapping is linear with
// slope 1/fSp; above it is logarithmic, joined so that both value and slope
// are continuous at the break point.
const (
	fSp       = 200.0 / 3.0 // Hz per mel in the linear region
	breakFreq = 1000.0      // linear/log transition, Hz
	breakMel  = breakFreq / fSp
)

// logStep is the natural log of the ratio spanned by the top 27 mel steps.
var logStep = math.Log(6.4) / 27.0

// HzToMel converts a frequency in Hz to mels on the given scale.
func HzToMel(f float64, s Scale) float64 {
	if s == HTK {
		return 2595.0 * math.Log10(1.0+f/700.0)
	}
	if f < breakFreq {
		return f / fSp
	}
	return breakMel + math.Log(f/breakFreq)/logStep
}

// MelToHz converts mels on the given scale to a frequency in Hz. It is the
// inverse of HzToMel.
func MelToHz(m float64, s Scale) float64 {
	if s == HTK {
		return 700.0 * (math.Pow(10.0, m/2595.0) - 1.0)
	}
	if m < breakMel {
		return m * fSp
	}
	return breakFreq * math.Exp(logStep*(m-breakMel))
}

// Norm selects filter normalisation for a mel filterbank.
type Norm int

const (
	// NormSlaney scales each filter by 2/(f_high - f_low) so that filters
	// are approximately constant energy rather than constant peak.
	// It corresponds to librosa's norm="slaney".
	NormSlaney Norm = iota

	// NormNone leaves filters at unit peak, corresponding to librosa's
	// norm=None.
	NormNone
)

// FilterBank builds an nMels x (nFFT/2+1) mel filterbank matrix, stored row
// major. Each row is a triangular filter whose vertices lie at three
// consecutive points spaced evenly on the mel scale between fMin and fMax.
//
// The result matches librosa.filters.mel for the corresponding arguments.
func FilterBank(sampleRate float64, nFFT, nMels int, fMin, fMax float64, s Scale, n Norm) []float64 {
	nBins := nFFT/2 + 1

	// Centre frequency of each FFT bin.
	binHz := make([]float64, nBins)
	for i := range binHz {
		binHz[i] = float64(i) * sampleRate / float64(nFFT)
	}

	// nMels+2 band edges, evenly spaced in mel.
	loMel := HzToMel(fMin, s)
	hiMel := HzToMel(fMax, s)
	edges := make([]float64, nMels+2)
	for i := range edges {
		m := loMel + (hiMel-loMel)*float64(i)/float64(nMels+1)
		edges[i] = MelToHz(m, s)
	}

	fb := make([]float64, nMels*nBins)
	for i := 0; i < nMels; i++ {
		lo, ctr, hi := edges[i], edges[i+1], edges[i+2]
		row := fb[i*nBins : (i+1)*nBins]
		for j, f := range binHz {
			// Rising and falling edges of the triangle. Taking the
			// smaller of the two and clamping at zero yields the
			// triangle without branching on which side we are.
			var up, down float64
			if ctr != lo {
				up = (f - lo) / (ctr - lo)
			}
			if hi != ctr {
				down = (hi - f) / (hi - ctr)
			}
			if v := math.Min(up, down); v > 0 {
				row[j] = v
			}
		}
		if n == NormSlaney {
			// Constant-energy rather than constant-peak scaling.
			enorm := 2.0 / (hi - lo)
			for j := range row {
				row[j] *= enorm
			}
		}
	}
	return fb
}
