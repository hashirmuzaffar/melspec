// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

// Package melspec computes mel spectrograms and mel-frequency cepstral
// coefficients that agree numerically with librosa.
//
// The package covers two primitives, MelSpectrogram and MFCC, and the pieces
// they are built from: a short-time Fourier transform, mel filterbanks on
// both the Slaney and HTK scales, and a power-to-decibel conversion. It is
// not a port of librosa; it is an independent implementation of the same
// published algorithms, verified against librosa output.
//
// # Layout
//
// Feature matrices are frame major: Rows is the number of frames and Cols
// the number of coefficients. This is the transpose of librosa's layout,
// which is coefficient major. Frame major keeps each frame's coefficients
// contiguous, which is what a streaming service iterating frame by frame
// wants.
//
// # Defaults
//
// Default and DefaultMFCC return librosa's defaults: 22050 Hz, a 2048 point
// FFT with a 512 sample hop, centred frames with zero padding, a periodic
// Hann window, 128 Slaney-scale mel bands with Slaney normalisation spanning
// 0 Hz to Nyquist, a power spectrogram, and 20 cepstral coefficients with no
// liftering.
//
// Two defaults are easy to get wrong when reimplementing:
//
//   - The decibel conversion inside MFCC uses a reference of 1, not the peak
//     of the input. Using the peak shifts every coefficient.
//   - The analysis window is the periodic Hann window, dividing by n rather
//     than n-1.
//
// # Agreement with librosa
//
// Measured against librosa 0.11.0, worst relative deviation over a broadband
// test signal:
//
//	short-time Fourier transform    2e-13
//	mel filterbank (float64)        3e-12
//	mel spectrogram                 5e-8
//	MFCC                            2e-7
//
// The transform and filterbank agree to double round-off. The mel and
// cepstral figures are larger because librosa builds its filterbank in
// float32 by default and casts up; this package uses float64 throughout, so
// its output is closer to the exact value than librosa's own default. See
// TestFilterBankExact, which pins the filterbank against librosa computed in
// float64.
package melspec
