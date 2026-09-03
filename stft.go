// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package melspec

import (
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/dsp/fourier"
)

// PadMode selects how the signal is extended when Center is true.
type PadMode int

const (
	// PadConstant extends the signal with zeros. This is librosa's default.
	PadConstant PadMode = iota
	// PadReflect mirrors the signal about its endpoints, excluding them.
	PadReflect
	// PadEdge repeats the first and last sample.
	PadEdge
)

// HannPeriodic returns an n point periodic Hann window,
//
//	w[k] = 0.5 - 0.5*cos(2*pi*k/n)
//
// The periodic form divides by n rather than n-1 and is the correct choice
// for spectral analysis; it is what scipy.signal.get_window returns with
// fftbins=True, and so what librosa uses.
func HannPeriodic(n int) []float64 {
	w := make([]float64, n)
	for k := range w {
		w[k] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(k)/float64(n))
	}
	return w
}

// padCenter returns w centred within a slice of length size, zero padded on
// both sides. It panics if size is smaller than len(w).
func padCenter(w []float64, size int) []float64 {
	if len(w) > size {
		panic("melspec: window longer than FFT size")
	}
	if len(w) == size {
		return w
	}
	out := make([]float64, size)
	copy(out[(size-len(w))/2:], w)
	return out
}

// pad extends x by n samples at each end using the given mode.
func pad(x []float64, n int, mode PadMode) []float64 {
	if n == 0 {
		return x
	}
	out := make([]float64, len(x)+2*n)
	copy(out[n:], x)
	switch mode {
	case PadConstant:
		// Zeros are already in place.
	case PadEdge:
		for i := 0; i < n; i++ {
			out[i] = x[0]
			out[len(out)-1-i] = x[len(x)-1]
		}
	case PadReflect:
		// Mirror about the endpoints without repeating them, matching
		// numpy.pad(mode="reflect").
		for i := 0; i < n; i++ {
			out[n-1-i] = x[reflectIndex(i+1, len(x))]
			out[n+len(x)+i] = x[reflectIndex(len(x)-2-i, len(x))]
		}
	}
	return out
}

// reflectIndex folds i into [0, n) by reflection about the endpoints.
func reflectIndex(i, n int) int {
	if n == 1 {
		return 0
	}
	period := 2 * (n - 1)
	i = ((i % period) + period) % period
	if i >= n {
		i = period - i
	}
	return i
}

// STFTOptions configures Spectrogram. The zero value is not useful; use
// DefaultSTFT and adjust.
type STFTOptions struct {
	NFFT      int       // FFT length.
	HopLength int       // Samples between successive frames.
	WinLength int       // Window length; zero means NFFT.
	Window    []float64 // Analysis window of length WinLength; nil means periodic Hann.
	Center    bool      // Pad by NFFT/2 so frame t is centred at t*HopLength.
	PadMode   PadMode   // Padding used when Center is true.
}

// DefaultSTFT returns librosa's default short-time Fourier transform
// settings: a 2048 point FFT, 512 sample hop, centred frames and constant
// (zero) padding.
func DefaultSTFT() STFTOptions {
	return STFTOptions{
		NFFT:      2048,
		HopLength: 512,
		Center:    true,
		PadMode:   PadConstant,
	}
}

// Frames reports the number of frames Spectrogram will produce for a signal
// of n samples.
func (o STFTOptions) Frames(n int) int {
	if o.Center {
		return 1 + n/o.HopLength
	}
	if n < o.NFFT {
		return 0
	}
	return 1 + (n-o.NFFT)/o.HopLength
}

// Spectrogram computes the magnitude short-time Fourier transform of x,
// raised to the given power. It returns a frames x (NFFT/2+1) matrix in row
// major order, so element (t, k) is at index t*(NFFT/2+1)+k.
//
// A power of 1 gives the magnitude spectrogram and 2 the power spectrogram.
func Spectrogram(x []float64, power float64, o STFTOptions) []float64 {
	win := o.Window
	if win == nil {
		n := o.WinLength
		if n == 0 {
			n = o.NFFT
		}
		win = HannPeriodic(n)
	}
	win = padCenter(win, o.NFFT)

	if o.Center {
		x = pad(x, o.NFFT/2, o.PadMode)
	}

	// After centre padding the frame count is 1+(len(x)-NFFT)/hop in both
	// cases, since padding added exactly NFFT samples.
	nBins := o.NFFT/2 + 1
	frames := 0
	if len(x) >= o.NFFT {
		frames = 1 + (len(x)-o.NFFT)/o.HopLength
	}

	out := make([]float64, frames*nBins)
	buf := make([]float64, o.NFFT)
	coef := make([]complex128, nBins)
	fft := fourier.NewFFT(o.NFFT)

	for t := 0; t < frames; t++ {
		seg := x[t*o.HopLength : t*o.HopLength+o.NFFT]
		for i, v := range seg {
			buf[i] = v * win[i]
		}
		coef = fft.Coefficients(coef, buf)
		row := out[t*nBins : (t+1)*nBins]
		for k, c := range coef {
			m := cmplx.Abs(c)
			switch power {
			case 1:
				row[k] = m
			case 2:
				row[k] = m * m
			default:
				row[k] = math.Pow(m, power)
			}
		}
	}
	return out
}
