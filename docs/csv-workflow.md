# Recording and reference comparison

## Two operating conditions

1. Track a video in **Video recording**. Verify acquisition FPS and uniform frame spacing; inspect the target and reference preview.
2. Download its motion CSV. Repeat for condition B. Record what changed and keep view geometry consistent.
3. Open **Compare recordings → Upload CSV files**, load A and B, select intended axes, and set a shared analysis band.
4. Review sample rates, durations, interval deviation, bin spacing and any band-edge warning. Export the Markdown report and JSON results.

Optical schema:

```csv
time_s,x_px,y_px
0.000000,0.000,0.000
0.033333,1.200,0.020
```

The short snippet illustrates the header only; analysis requires at least 32 rows. Use the app's synthetic CSV downloads for complete examples.

## MPU6050 reference

Use **Camera vs sensor** with an optical motion CSV and a sensor CSV:

```csv
time_s,ax_m_s2,ay_m_s2,az_m_s2
0.000000,0.120,0.010,9.810
0.005000,0.180,0.020,9.820
```

Firmware must log device acquisition timestamps in **seconds**, acceleration in **m/s²**, and all three axes. Convert device counts using the configured full-scale sensitivity; convert g to m/s² with an explicitly documented convention (commonly 9.80665 m/s² per g). Record sensor sample rate, digital filter settings, full scale, mounting and axis orientation. Host packet-arrival times are not acquisition timestamps. CSV import does not configure or validate firmware.

Compare physically corresponding axes and locations under the same operating condition. The sensors need not share a clock for independent stationary-frequency estimates. This workflow does **not** synchronize signals or evaluate phase/coherence. Mounting mass and sensor noise can alter or obscure the motion.

## Input policy

- UTF-8 CSV, exact column names, finite numeric values, 32–200,000 rows and at most 10 MB.
- Strictly increasing timestamps. No sorting, clock-reset repair, duplicate removal or missing-row removal.
- Each interval must be within 2% of the mean interval. Larger jitter or a missing-sample gap is rejected. Small accepted jitter is linearly resampled onto an equal grid and disclosed in the report. This threshold is a prototype policy, not a validated accuracy specification.
- A shared band may extend to at most 45% of the slower sample rate and must contain three FFT bins from each recording. The guard band does not prevent or detect aliasing already present.

## Interpret results

Each channel is linearly detrended and analyzed with a Hann-windowed periodogram. The largest eligible bin is selected separately inside the shared band. Bin spacing depends on recording length and is not an error bar. Very small or noise-dominated signals can still have a largest spectral peak.

Spectral curves are normalized independently to their own band maxima. This compares shape, not amplitude. Acceleration weights high-frequency components more strongly than displacement, so dominant peaks can differ even for the same physical motion. The report deliberately avoids amplitude ratios across incompatible units or uncalibrated camera views.

No agreement tolerance, fault label, maintenance alarm or industrial accuracy claim is inferred. Synthetic examples validate software behavior only. Physical repeatability and reference measurements remain required.
