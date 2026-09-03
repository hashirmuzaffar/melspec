// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package melspec

import "testing"

func benchSignal(n int) []float64 {
	y := make([]float64, n)
	var s float64 = 0.5
	for i := range y {
		// Cheap deterministic filler; content does not affect timing.
		s = s*1.0000001 + 1e-9
		y[i] = s - float64(int(s))
	}
	return y
}

func BenchmarkMelSpectrogram1s(b *testing.B) {
	y := benchSignal(22050)
	o := Default()
	b.ReportAllocs()
	for b.Loop() {
		MelSpectrogram(y, o)
	}
}

func BenchmarkMFCC1s(b *testing.B) {
	y := benchSignal(22050)
	o, m := Default(), DefaultMFCC()
	b.ReportAllocs()
	for b.Loop() {
		MFCC(y, o, m)
	}
}

func BenchmarkFilterBank(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		FilterBank(22050, 2048, 128, 0, 11025, Slaney, NormSlaney)
	}
}
