// Copyright 2026 Hashir Muzaffar. All rights reserved.
// Use of this source code is governed by a BSD-3-Clause license
// that can be found in the LICENSE file.

package melspec

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Tolerances for comparison against librosa. librosa builds its mel
// filterbank in float32 by default and casts up, so mel-domain results carry
// a relative error of order 1e-7 that is inherent to the reference rather
// than to this implementation. See TestFilterBankFloat32Origin.
const (
	rtol = 1e-6
	atol = 1e-9

	// Decibel-domain results cross zero, so they need an absolute bound.
	// Agreement with librosa's defaults is limited to about 3e-7 dB by
	// librosa's float32 filterbank, not by this package; see
	// TestFilterBankExact.
	atolDB = 1e-5
)

func load(t *testing.T, name string) Matrix {
	t.Helper()
	f, err := os.Open(filepath.Join("internal", "testdata", name+".txt"))
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	if !sc.Scan() {
		t.Fatalf("%s: empty fixture", name)
	}
	var rows, cols int
	if _, err := fmt.Sscanf(sc.Text(), "%d %d", &rows, &cols); err != nil {
		t.Fatalf("%s: bad header: %v", name, err)
	}

	m := Matrix{Data: make([]float64, 0, rows*cols), Rows: rows, Cols: cols}
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		for _, tok := range strings.Fields(line) {
			v, err := strconv.ParseFloat(tok, 64)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			m.Data = append(m.Data, v)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if len(m.Data) != rows*cols {
		t.Fatalf("%s: got %d values, want %d", name, len(m.Data), rows*cols)
	}
	return m
}

func signal(t *testing.T) []float64 {
	t.Helper()
	return load(t, "signal").Data
}

// compare reports the worst absolute and relative deviation between got and
// want, failing if any element exceeds the tolerance.
func compare(t *testing.T, name string, got, want Matrix, rtol, atol float64) {
	t.Helper()
	if got.Rows != want.Rows || got.Cols != want.Cols {
		t.Fatalf("%s: shape = %dx%d, want %dx%d",
			name, got.Rows, got.Cols, want.Rows, want.Cols)
	}
	var worstAbs, worstRel float64
	var bad int
	for i, w := range want.Data {
		g := got.Data[i]
		if math.IsNaN(g) != math.IsNaN(w) {
			t.Fatalf("%s: element %d NaN mismatch: got %v want %v", name, i, g, w)
		}
		d := math.Abs(g - w)
		worstAbs = math.Max(worstAbs, d)
		if a := math.Abs(w); a > 0 {
			worstRel = math.Max(worstRel, d/a)
		}
		if d > atol+rtol*math.Abs(w) {
			bad++
			if bad <= 3 {
				t.Errorf("%s: element %d = %.17g, want %.17g (Δ %.3g)", name, i, g, w, d)
			}
		}
	}
	if bad != 0 {
		t.Errorf("%s: %d/%d elements outside tolerance", name, bad, len(want.Data))
	}
	t.Logf("%-18s %4dx%-5d worst abs %.3e  worst rel %.3e",
		name, got.Rows, got.Cols, worstAbs, worstRel)
}

func TestFilterBank(t *testing.T) {
	for _, tc := range []struct {
		name       string
		nMels      int
		fMin, fMax float64
		scale      Scale
		norm       Norm
	}{
		{"fb_slaney", 128, 0, 0, Slaney, NormSlaney},
		{"fb_htk", 128, 0, 0, HTK, NormSlaney},
		{"fb_nonorm", 128, 0, 0, Slaney, NormNone},
		{"fb_40_300_8000", 40, 300, 8000, Slaney, NormSlaney},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := load(t, tc.name) // stored as bins x mels
			const nFFT = 2048
			nBins := nFFT/2 + 1
			fMax := tc.fMax
			if fMax == 0 {
				fMax = 22050.0 / 2
			}
			fb := FilterBank(22050, nFFT, tc.nMels, tc.fMin, fMax, tc.scale, tc.norm)

			// Transpose to the fixture's bins x mels layout.
			got := Matrix{Data: make([]float64, nBins*tc.nMels), Rows: nBins, Cols: tc.nMels}
			for i := 0; i < tc.nMels; i++ {
				for j := 0; j < nBins; j++ {
					got.Data[j*tc.nMels+i] = fb[i*nBins+j]
				}
			}
			compare(t, tc.name, got, want, rtol, atol)
		})
	}
}

func TestSpectrogram(t *testing.T) {
	y := signal(t)
	for _, tc := range []struct {
		name   string
		center bool
		mode   PadMode
	}{
		{"stft_mag", true, PadConstant},
		{"stft_mag_reflect", true, PadReflect},
		{"stft_mag_nocenter", false, PadConstant},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := load(t, tc.name)
			o := DefaultSTFT()
			o.Center = tc.center
			o.PadMode = tc.mode
			spec := Spectrogram(y, 1, o)
			nBins := o.NFFT/2 + 1
			got := Matrix{Data: spec, Rows: len(spec) / nBins, Cols: nBins}
			compare(t, tc.name, got, want, rtol, atol)
		})
	}
}

func TestMelSpectrogram(t *testing.T) {
	y := signal(t)
	for _, tc := range []struct {
		name   string
		mutate func(*Options)
	}{
		{"mel_default", func(o *Options) {}},
		{"mel_htk", func(o *Options) { o.Scale = HTK }},
		{"mel_power1", func(o *Options) { o.Power = 1 }},
		{"mel_40_300_8000", func(o *Options) { o.NMels, o.FMin, o.FMax = 40, 300, 8000 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := load(t, tc.name)
			o := Default()
			tc.mutate(&o)
			compare(t, tc.name, MelSpectrogram(y, o), want, rtol, atol)
		})
	}
}

func TestPowerToDB(t *testing.T) {
	y := signal(t)
	want := load(t, "powerdb_default")
	mel := MelSpectrogram(y, Default())
	PowerToDB(mel.Data, DefaultDB())
	compare(t, "powerdb_default", mel, want, rtol, atolDB)
}

func TestMFCC(t *testing.T) {
	y := signal(t)
	for _, tc := range []struct {
		name   string
		mutate func(*MFCCOptions)
	}{
		{"mfcc_default", func(m *MFCCOptions) {}},
		{"mfcc_13", func(m *MFCCOptions) { m.N = 13 }},
		{"mfcc_lifter22", func(m *MFCCOptions) { m.Lifter = 22 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := load(t, tc.name)
			m := DefaultMFCC()
			tc.mutate(&m)
			compare(t, tc.name, MFCC(y, Default(), m), want, rtol, atolDB)
		})
	}
}

// TestFilterBankExact checks the filterbank against librosa computed in
// float64. Agreement here is at the level of double round-off, which
// establishes that the mel scale, band edges and triangle construction are
// exact and that any larger disagreement elsewhere comes from librosa's
// float32 default rather than from this package.
func TestFilterBankExact(t *testing.T) {
	want := load(t, "fb_slaney_f64") // bins x mels
	const nFFT = 2048
	nBins := nFFT/2 + 1
	fb := FilterBank(22050, nFFT, 128, 0, 11025, Slaney, NormSlaney)

	var worstAbs, worstRel float64
	for i := 0; i < 128; i++ {
		for j := 0; j < nBins; j++ {
			d := math.Abs(fb[i*nBins+j] - want.At(j, i))
			worstAbs = math.Max(worstAbs, d)
			if w := math.Abs(want.At(j, i)); w != 0 {
				worstRel = math.Max(worstRel, d/w)
			}
		}
	}
	t.Logf("filterbank vs librosa float64: worst abs %.3e  rel %.3e", worstAbs, worstRel)
	if worstRel > 1e-11 {
		t.Errorf("filterbank relative error %.3e exceeds 1e-11", worstRel)
	}
}

// TestMelBandEdges checks the mel scale round trip and the band edge
// placement against librosa.mel_frequencies.
func TestMelBandEdges(t *testing.T) {
	want := load(t, "mel_edges_slaney")
	loMel, hiMel := HzToMel(0, Slaney), HzToMel(11025, Slaney)
	var worst float64
	for i := 0; i < want.Rows; i++ {
		got := MelToHz(loMel+(hiMel-loMel)*float64(i)/float64(want.Rows-1), Slaney)
		worst = math.Max(worst, math.Abs(got-want.At(i, 0)))
	}
	t.Logf("mel band edges: worst abs %.3e Hz", worst)
	if worst > 1e-9 {
		t.Errorf("band edge error %.3e Hz exceeds 1e-9", worst)
	}

	// Round trip on both scales.
	for _, s := range []Scale{Slaney, HTK} {
		for f := 0.0; f <= 11025; f += 37.5 {
			if got := MelToHz(HzToMel(f, s), s); math.Abs(got-f) > 1e-9 {
				t.Fatalf("scale %v: round trip of %g Hz gave %g", s, f, got)
			}
		}
	}
}
