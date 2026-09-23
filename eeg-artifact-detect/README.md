# eeg_autoclean

Non-destructive, four-layer artifact detection for EEG recordings. It scans a
recording for eye-blink, muscle, cardiac, and gross amplitude/flat-channel
artifacts and returns annotations describing what it found and where -- it
never modifies your data unless you explicitly ask it to (`apply_cleaning`).

Built on [MNE-Python](https://mne.tools/).

## Why

Most EEG preprocessing pipelines either hand-tune fixed microvolt thresholds
per dataset, or reach straight for a full ICA-based cleaning pass with no
lighter-weight option in between. This project's angle: adaptive per-channel
thresholds that scale to each channel's own noise floor (no per-dataset
tuning), four independent detection mechanisms instead of one, and every
detection step kept separate from any decision about what to do with it.
Detection and cleaning are two different opt-in steps, not one bundled
action.

## Installation

No virtual environment required, but one is recommended.

```bash
git clone <repo-url>
cd eeg-artifact-detect
pip install -e .
```

Consensus mode (cross-checking against AutoReject) needs an extra, optional
dependency:

```bash
pip install -e ".[consensus]"
```

Running the test suite:

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Quick start

```python
from eeg_autoclean import load_eeg, detect_artifacts, plot_artifacts

raw = load_eeg("recording.edf")
result = detect_artifacts(raw)

print(result["annotations"])          # BAD_amplitude / BAD_flat / BAD_muscle / BAD_cardiac
print(result["ica_components_flagged"])  # ocular ICA components, or None if no EOG channel

plot_artifacts(raw, result, output_path="artifacts.png")
```

`detect_artifacts` never mutates `raw` or drops anything -- it returns
`mne.Annotations` describing what it found, in the same spirit as MNE's own
`raw.annotations`. Attach them yourself (`raw.set_annotations(result["annotations"])`)
or pass them straight into whatever you already use `reject_by_annotation` for.

If you do want a cleaned copy back:

```python
from eeg_autoclean import apply_cleaning

cleaned, report = apply_cleaning(
    raw, result,
    remove_ocular_ica=True,
    drop_bad_channels=True,
    crop_bad_segments=True,
)
print(report)  # what was actually removed/interpolated/cropped, and how much
```

## Architecture

Four layers, each looking for a structurally different kind of artifact, all
merged into one `mne.Annotations` object (Layer 2 aside, which flags ICA
components rather than time windows):

**Layer 1 -- amplitude / flat.** Per-channel adaptive threshold: median plus
`k * MAD` (median absolute deviation, scaled to behave like a robust
standard deviation) of that channel's own peak-to-peak amplitude across all
windows.
This scales to each channel's actual noise floor instead of one fixed
microvolt value across every channel and recording, and stays reliable even
when a chunk of a channel's own windows are themselves artifacts (median/MAD
only break down once more than half the windows are outliers). Flags
`BAD_amplitude` (too high) and `BAD_flat` (near-zero variance -- a
disconnected or bridged channel).

**Layer 2 -- ocular (ICA).** Fits ICA on the EEG channels; if an EOG channel
is present, correlates each component against it two ways -- against
blink-locked epochs, and against the whole continuous recording (catching
non-blink ocular activity like saccades) -- and flags the union. Skipped
gracefully if there's no EOG channel to correlate against.

**Layer 3 -- muscle.** Per-channel power ratio between a muscle band
(20-40 Hz) and a reference band (1-20 Hz), computed with Welch's method, on
the same adaptive median + k * MAD threshold as Layer 1. EMG contamination is
broadband and skews high-frequency relative to the reference band, which is
what this ratio picks up on.

**Layer 4 -- cardiac.** Per-channel autocorrelation in a plausible heart-rate
range (40-140 bpm), on a longer window than Layers 1/3 since periodicity
needs several full pulse cycles to show up. If an ECG channel is present, a
second, independent detection path cross-correlates each EEG channel against
it directly. A flag only survives if it persists across 2 consecutive
windows -- real cardiac contamination is sustained, periodic activity, not a
single isolated burst, and requiring persistence cuts false positives
substantially without costing recall (see Validation below).

Beyond the four layers:

- **`detect_eye_state(raw)`** -- classifies a recording as eyes-open or
  eyes-closed from the posterior alpha "bump" (8-12 Hz power relative to its
  flanking bands) over O1/O2/Oz-equivalent channels.
- **`compute_quality_score(raw, result)`** -- combines all four layers into a
  single 0-100 score (100 = nothing flagged anywhere), with a transparent
  per-layer breakdown so the single number is never the only thing you get.
  Equal-weighted by default; pass `weights` to reflect your own priorities
  (an ERP study cares differently about ocular contamination than a sleep
  study cares about muscle).
- **`apply_cleaning(raw, result, ...)`** -- the opt-in destructive step: ICA
  component removal, bad-channel exclude/interpolate, and bad-segment
  cropping, each off by default and selected independently.
- **`process_directory(dir_path)`** -- runs detection + quality scoring
  across every `.edf`/`.bdf` file in a directory and writes a summary CSV.
  One bad file never aborts the batch; failures are caught and reported as a
  row, not raised.
- **`run_consensus(raw)`** -- cross-checks our detector against
  [AutoReject](https://autoreject.github.io/) on the same time/channel grid.
  See Validation below for exactly what this does and doesn't show.

## Validation

### Synthetic ground truth

`scripts/generate_synthetic.py` builds synthetic recordings with known
injected artifacts across four severity scenarios (sparse/mild, moderate,
dense/severe, noisy background) and scores each layer against ground truth.
Mean precision/recall/F1 across all four scenarios, current defaults:

| Layer | Precision | Recall | F1 |
|---|---|---|---|
| Layer 1 -- amplitude | 1.00 | 1.00 | 1.00 |
| Layer 1 -- flat | 1.00 | 0.97 | 0.98 |
| Layer 2 -- ocular (ICA) | 0.75 | 0.75 | 0.75 |
| Layer 3 -- muscle | 0.48 | 0.78 | 0.57 |
| Layer 4 -- cardiac | 0.67 | 0.75 | 0.70 |

Layer 1 is close to a solved problem on this synthetic battery -- the
adaptive threshold does what it's supposed to. Layers 3 and 4 trade
precision for recall by design: both use the same median + k * MAD adaptive
threshold as Layer 1, but their underlying metrics (a spectral power ratio,
an autocorrelation coefficient) are noisier than raw peak-to-peak amplitude,
so more false positives get through at a sensitivity that still catches most
real events. `eeg_autoclean/muscle.py` and `eeg_autoclean/cardiac.py`
document the parameter sweeps (window size, Welch `nperseg`, `mad_k`,
persistence) that were tried before settling on the current defaults.

Layer 4's persistence requirement (`cardiac_persistence_n=2`) specifically:

```
persistence_n=1: mean P=0.29 (sd=0.17)  R=0.75 (sd=0.43)  F1=0.41 (sd=0.24)
persistence_n=2: mean P=0.67 (sd=0.41)  R=0.75 (sd=0.43)  F1=0.70 (sd=0.41)
```

Precision more than doubles and recall is unchanged -- identical per
scenario (1.00/1.00/1.00/0.00 either way), not just on average. Every real
detection that existed at `persistence_n=1` survives; only false positives
are suppressed. The one scenario stuck at 0.0 recall ("noisy background")
is unaffected by persistence either way -- background noise there overwhelms
the pulse amplitude itself, a signal-to-noise floor persistence filtering
can't do anything about.

Layer 2's soft spot is the same "noisy background" scenario -- high
background noise degrades the ICA decomposition enough that ocular
components stop separating out cleanly, independent of anything Layer 4 is
doing.

### Real datasets

`scripts/validate_real_datasets.py` runs the detector against three public
MNE-bundled datasets with no synthetic ground truth, as a structural sanity
check (does it behave reasonably, not "is this the right F1"):

| Dataset | Amplitude windows | Flat windows | Layer 2 vs `find_bads_eog` | Muscle windows (sanity check) |
|---|---|---|---|---|
| `sample` | 12/60 | 0/60 | 1/1 components matched | 5/60 |
| `erp_core` | 15/61 | 0/61 | 1/1 components matched | 5/61 |
| `sleep_physionet` | 1/301 | 0/301 | n/a -- MNE flagged nothing to compare against | 25/301 |

Layer 2's agreement check compares our flagged ICA components against MNE's
own `ica.find_bads_eog(raw)` on the real EOG channel -- exact match on both
datasets that have one. There's no ground truth for Layer 3 on real data, so
that column is just "how often did it fire," not a precision claim.

`docs/images/sample_validation.png` (generated by
`scripts/validate_on_sample.py`) shows all three time-window layers on a
20-second segment of the MNE sample dataset.

### Baseline comparison

`scripts/compare_baseline.py` asks a narrower, more useful question than "is
this tool good": does anything here earn its place over the simplest
possible alternative -- fit ICA once, call `ica.find_bads_eog(raw)` on the
continuous recording, done? Same fitted ICA and the same `find_bads_eog`
call our own whole-recording pass already makes internally, reused rather
than refit, so ICA-fit randomness can't blur the comparison.

Two honest findings came out of this:

1. **Layer 1 catches things no ICA+EOG approach can, even in principle.**
   Across the synthetic battery and all three real datasets, Layer 1 flags
   gross amplitude spikes and flat/disconnected channels that aren't ocular
   at all -- a baseline with no amplitude scan produces zero of these, by
   construction. Cross-checked directly: the max correlation between any
   flagged ocular component and a gross-artifact indicator waveform stays
   near 0 across every synthetic scenario, confirming the components aren't
   accidentally picking up the gross artifacts either.
2. **The blink-epoch correlation pass currently adds nothing measurable.**
   Layer 2 runs two passes -- correlate against blink-locked epochs, and
   correlate against the whole continuous recording -- and unions the
   result. Across all 7 test cases (4 synthetic scenarios + 3 real
   datasets), the blink-epoch pass never flagged a component the
   whole-recording pass didn't already catch. It was originally added
   because it helped before the whole-recording pass existed; now that both
   run together, it looks redundant on this battery. Kept in as a cheap
   safeguard rather than removed, but that's a hypothesis, not a
   demonstrated benefit -- stated here plainly rather than assumed.

### Consensus mode

`run_consensus()` cross-checks our detector against AutoReject on the same
(channel, window) grid and buckets every cell as `AGREE_BAD`, `AGREE_CLEAN`,
or `DISAGREE`. `scripts/validate_consensus.py` runs this on the synthetic
battery and the real `sample` dataset. The two methods rest on different
assumptions -- ours: per-layer amplitude/spectral/periodicity signatures;
AutoReject: cross-validated per-epoch/per-channel statistical thresholds --
so they have different blind spots, and the interesting question is which
disagreements are actually meaningful.

**Validated:** cells we flag that AutoReject doesn't are reliably real
artifacts. 21-29 of each synthetic scenario's ours-only disagreements land on
a known ground-truth cell (29/31 in "moderate", 21/24 in "dense/severe"),
versus only 1-3 that don't, in every scenario. AutoReject's amplitude/
covariance-based method structurally can't see the spectral/periodicity
signatures Layers 3 and 4 are built to catch, and this result confirms that
gap is real rather than incidental.

**Not validated:** cells AutoReject flags that we don't, which numerically
dominate the DISAGREE category (roughly 280-350 per scenario vs. 5-31
ours-only). Across all four synthetic scenarios, an autoreject-only cell
matched a known ground-truth artifact **zero** times. That doesn't mean
AutoReject is wrong -- it's a broader, more conservative method not scoped
to this project's specific artifact taxonomy, and may be catching genuine
statistical outliers outside it -- but it does mean this project has no
evidence that autoreject-only disagreements are worth prioritizing.
AutoReject's overall flag rate here runs roughly 5-10x ours (checked at both
1.0s and 2.0s epochs, so not an artifact of epoch length), and that gap is
open territory: not confirmed as noise, not confirmed as signal.

Cell counts, synthetic battery:

| Scenario | AGREE_BAD | AGREE_CLEAN | DISAGREE | ours-only (real) | autoreject-only (real) |
|---|---|---|---|---|---|
| sparse/mild | 26 (2.7%) | 622 (64.8%) | 312 (32.5%) | 5 (2) | 307 (0) |
| moderate | 15 (1.6%) | 566 (59.0%) | 379 (39.5%) | 31 (29) | 348 (0) |
| dense/severe | 41 (4.3%) | 615 (64.1%) | 304 (31.7%) | 24 (21) | 280 (0) |
| noisy background | 5 (0.5%) | 617 (64.3%) | 338 (35.2%) | 7 (6) | 331 (0) |

On the real `sample` dataset (3600 cells total): 227 AGREE_BAD (6.3%), 2764
AGREE_CLEAN (76.8%), 609 DISAGREE (16.9%). A spot-check of the first 10
DISAGREE cells found all of them `flagged_by=autoreject`, with no visually
obvious artifact in a manual check of a small sample of them -- consistent
with, though not proof of, the synthetic-battery finding above.

Consensus mode requires channel positions (a montage) on `raw`, since
AutoReject needs these for its spatial interpolation-based reasoning;
`run_consensus()` raises a clear error up front if none is set, rather than
surfacing AutoReject's own internal error.

## Known limitations

- **Layer 2 needs enough EEG channels to work at all.** ICA needs a
  reasonable number of channels to separate ocular activity into its own
  component cleanly; it's unreliable on very low channel counts (e.g. the
  2-channel `sleep_physionet` dataset above has nothing for MNE's own
  `find_bads_eog` to compare against either).
- **Layers 3 and 4 trade precision for recall.** Both are validated at
  P < 0.7 on the synthetic battery -- expect more false positives than
  Layer 1, by design of the underlying metrics rather than a tuning miss.
  See Validation above for the actual numbers and the parameter sweeps
  behind the current defaults.
- **The "noisy background" synthetic scenario is a shared soft spot** for
  both Layer 2 (ICA decomposition degrades) and Layer 4 (pulse signal
  overwhelmed by background noise, 0.0 recall regardless of persistence
  setting).
- **The blink-epoch ICA pass isn't earning its keep on the current test
  battery** (see Baseline comparison above) -- kept in as a cheap safeguard,
  not because it's been shown to help.
- **`crop_bad_segments` can crop disproportionately because of Layer 4.**
  `BAD_cardiac` spans are 6 seconds each (vs. 1 second for Layer 1/3) and
  Layer 4 has the lowest precision of the four layers, so a handful of false
  positives can remove a large fraction of a recording (one dense/severe
  test run: 6 flagged cardiac windows cropped 47 of 60 seconds). Pass a
  narrower `crop_descriptions` if this is too aggressive for your use case.
- **`plot_artifacts()` doesn't shade `BAD_cardiac` regions** -- its color
  map currently only covers `BAD_amplitude`, `BAD_flat`, and `BAD_muscle`,
  so Layer 4 flags exist in the returned annotations but aren't visualized.
- **Consensus mode's autoreject-only disagreements are unexplained**, not
  just unvalidated -- see Consensus mode above. Don't treat them as a
  prioritized "review these first" list; there's no evidence for that yet.
- **Consensus mode excludes Layer 2** (ICA components aren't a per-cell
  time/channel flag the way Layer 1/3/4's annotations are, so there's no
  direct way to put them on the same grid as AutoReject's per-epoch
  judgment without an extra projection step this version doesn't do).
- Only `.edf`/`.bdf` input is supported directly (`load_eeg`); anything MNE
  can already load into a `Raw` object works with `detect_artifacts` itself.

## License

MIT
