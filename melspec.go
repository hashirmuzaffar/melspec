// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package melspec

// Matrix is a dense row-major matrix. Element (r, c) is Data[r*Cols+c].
//
// Feature matrices produced by this package are frame major: Rows is the
// number of frames and Cols the number of coefficients. Note that this is
// the transpose of librosa's layout, which is coefficient major.
type Matrix struct {
	Data       []float64
	Rows, Cols int
}

// At returns element (r, c).
func (m Matrix) At(r, c int) float64 { return m.Data[r*m.Cols+c] }

// Row returns a view of row r. The result aliases m.Data.
func (m Matrix) Row(r int) []float64 { return m.Data[r*m.Cols : (r+1)*m.Cols] }

// Options configures MelSpectrogram and MFCC.
type Options struct {
	SampleRate float64     // Sample rate of the input signal in Hz.
	STFT       STFTOptions // Short-time Fourier transform settings.
	Power      float64     // Exponent on the magnitude spectrum; 2 gives power.
	NMels      int         // Number of mel bands.
	FMin       float64     // Lowest band edge in Hz.
	FMax       float64     // Highest band edge in Hz; zero means SampleRate/2.
	Scale      Scale       // Mel warping.
	Norm       Norm        // Filter normalisation.
}

// Default returns options matching librosa's defaults: 22050 Hz, a 2048
// point FFT with 512 sample hop, centred frames with zero padding, a power
// spectrogram, and 128 Slaney-scale mel bands with Slaney normalisation
// spanning 0 Hz to the Nyquist frequency.
func Default() Options {
	return Options{
		SampleRate: 22050,
		STFT:       DefaultSTFT(),
		Power:      2,
		NMels:      128,
		FMin:       0,
		FMax:       0,
		Scale:      Slaney,
		Norm:       NormSlaney,
	}
}

func (o Options) fMax() float64 {
	if o.FMax != 0 {
		return o.FMax
	}
	return o.SampleRate / 2
}

// MelSpectrogram computes the mel-scaled spectrogram of x. The result has one
// row per frame and o.NMels columns.
func MelSpectrogram(x []float64, o Options) Matrix {
	spec := Spectrogram(x, o.Power, o.STFT)
	nBins := o.STFT.NFFT/2 + 1
	frames := 0
	if nBins > 0 {
		frames = len(spec) / nBins
	}

	fb := FilterBank(o.SampleRate, o.STFT.NFFT, o.NMels, o.FMin, o.fMax(), o.Scale, o.Norm)

	out := make([]float64, frames*o.NMels)
	for t := 0; t < frames; t++ {
		frame := spec[t*nBins : (t+1)*nBins]
		row := out[t*o.NMels : (t+1)*o.NMels]
		for i := 0; i < o.NMels; i++ {
			filt := fb[i*nBins : (i+1)*nBins]
			var sum float64
			for k, w := range filt {
				sum += w * frame[k]
			}
			row[i] = sum
		}
	}
	return Matrix{Data: out, Rows: frames, Cols: o.NMels}
}

// BandCenters returns the centre frequency in Hz of each of the o.NMels mel
// bands. Band i is the triangle whose peak sits at BandCenters(o)[i], which
// makes it the natural axis label for a mel spectrogram.
func BandCenters(o Options) []float64 {
	loMel := HzToMel(o.FMin, o.Scale)
	hiMel := HzToMel(o.fMax(), o.Scale)
	c := make([]float64, o.NMels)
	for i := range c {
		m := loMel + (hiMel-loMel)*float64(i+1)/float64(o.NMels+1)
		c[i] = MelToHz(m, o.Scale)
	}
	return c
}
