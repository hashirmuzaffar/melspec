// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package melspec_test

import (
	"fmt"
	"math"

	"github.com/hashirmuzaffar/melspec"
)

func ExampleMelSpectrogram() {
	// One second of a 440 Hz tone at 22050 Hz.
	sr := 22050.0
	y := make([]float64, int(sr))
	for i := range y {
		y[i] = math.Sin(2 * math.Pi * 440 * float64(i) / sr)
	}

	mel := melspec.MelSpectrogram(y, melspec.Default())
	fmt.Printf("%d frames x %d mel bands\n", mel.Rows, mel.Cols)

	// The loudest band of the middle frame should sit near 440 Hz.
	row := mel.Row(mel.Rows / 2)
	peak := 0
	for i, v := range row {
		if v > row[peak] {
			peak = i
		}
	}
	fmt.Printf("peak band %d, centre %.0f Hz\n", peak, melspec.BandCenters(melspec.Default())[peak])

	// Output:
	// 44 frames x 128 mel bands
	// peak band 16, centre 438 Hz
}

func ExampleMFCC() {
	sr := 22050.0
	y := make([]float64, int(sr)/2)
	for i := range y {
		y[i] = math.Sin(2*math.Pi*220*float64(i)/sr) +
			0.5*math.Sin(2*math.Pi*1760*float64(i)/sr)
	}

	m := melspec.MFCC(y, melspec.Default(), melspec.DefaultMFCC())
	fmt.Printf("%d frames x %d coefficients\n", m.Rows, m.Cols)

	// Output:
	// 22 frames x 20 coefficients
}
