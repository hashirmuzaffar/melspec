# melspec

Mel spectrograms and MFCCs for Go, with numerical parity against
[librosa](https://librosa.org).

Go has FFTs, and `gonum.org/v1/gonum/dsp/fourier` is solid, but nothing that
produces the two features every speech and audio model actually consumes. So
Go services end up shelling out to Python, or reimplementing the feature
extractor and quietly disagreeing with the model's training pipeline. This
package is the missing primitive, and it is tested against librosa rather
than against itself.

It is not a port. It is an independent implementation of the same published
algorithms: the Slaney and HTK mel scales, triangular filterbanks, an
orthonormal DCT-II, verified numerically against librosa's output.

```go
import "github.com/hashirmuzaffar/melspec"

mel := melspec.MelSpectrogram(samples, melspec.Default())
m   := melspec.MFCC(samples, melspec.Default(), melspec.DefaultMFCC())

fmt.Println(mel.Rows, mel.Cols)  // frames x 128
fmt.Println(m.Rows, m.Cols)      // frames x 20
```

`Default()` and `DefaultMFCC()` reproduce librosa's defaults: 22050 Hz, a
2048-point FFT with 512-sample hop, centred frames with zero padding, a
periodic Hann window, 128 Slaney-scale mel bands with Slaney normalisation
from 0 Hz to Nyquist, a power spectrogram, and 20 cepstral coefficients
without liftering.

## Agreement with librosa

Worst relative deviation from librosa 0.11.0 over a broadband test signal:

| | worst relative | worst absolute |
|---|---:|---:|
| short-time Fourier transform | 2.0e-13 | 4.3e-14 |
| mel filterbank vs librosa float64 | 3.1e-12 | 3.5e-16 |
| mel band edges | n/a | 1.8e-12 Hz |
| mel spectrogram | 5.1e-8 | n/a |
| MFCC | 2.3e-7 | 1.9e-7 |
| power to dB | n/a | 2.2e-7 dB |

The transform and the filterbank agree to double round-off. The mel and
cepstral figures are larger for a reason worth stating plainly: **librosa
builds its mel filterbank in `float32` by default** and casts up. This
package works in `float64` throughout, so where the two disagree at the 1e-7
level, this package is the more accurate of the two. `TestFilterBankExact`
pins the filterbank against librosa computed in `float64`, where agreement
returns to round-off.

Covered by the parity suite: both mel scales, both normalisations, all three
pad modes, `center` on and off, non-default band counts and frequency
ranges, `power=1` and `power=2`, several coefficient counts, and liftering.

## Two defaults that are easy to get wrong

Both are verified in the test suite rather than assumed.

**`MFCC` converts to decibels against a reference of 1, not the peak.**
librosa's `power_to_db` defaults to `ref=1.0`, and `librosa.feature.mfcc`
uses that default. Substituting `ref=np.max`, which many reimplementations
do, moves every coefficient, by 293 dB on the test signal.

**The analysis window is the *periodic* Hann window**, dividing by `n` rather
than `n-1`. That is what `scipy.signal.get_window(..., fftbins=True)`
returns and what librosa uses.

## Regenerating the fixtures

Ground truth lives in `internal/testdata` as plain text. To regenerate it
against a different librosa version:

```
pip install librosa
python cmd/genfixtures/gen.py
go test ./...
```

## Layout

Feature matrices are frame major: `Rows` is the frame count, `Cols` the
coefficient count. This is the transpose of librosa's layout. Frame major
keeps each frame contiguous, which is what a service iterating frame by
frame wants.

## Performance

On an M-series Mac, one second of 22050 Hz audio:

```
BenchmarkMelSpectrogram1s     5.5 ms/op     11 allocs/op
BenchmarkMFCC1s               5.4 ms/op     13 allocs/op
BenchmarkFilterBank           0.3 ms/op      3 allocs/op
```

Roughly 180x realtime. Note that both entry points rebuild the filterbank
and the FFT plan on every call. For a service extracting features
continuously, a reusable analyser holding both would remove that overhead;
that is the obvious next addition and is not here yet.

## Status

Covers `MelSpectrogram` and `MFCC` and the pieces beneath them. Deliberately
narrow. Not implemented: delta features, CQT, chroma, spectral contrast,
inverse transforms, or audio file decoding. Bring your own decoder and hand
this package `[]float64`.

## Licence

BSD-3-Clause.
