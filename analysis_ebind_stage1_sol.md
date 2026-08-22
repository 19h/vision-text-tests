# E-BIND-1 repeated-measures analysis

Rates are means of per-item call probabilities; intervals resample pages then items. All-zero/all-one endpoints use a boundary-safe Wilson interval over top-level clusters.

| carrier/encoding | items | calls | C | W | N | W0 | clustering |
|---|---:|---:|---:|---:|---:|---:|---|
| image/hex | 120 | 360 | 0.633 [0.533, 0.725] | 0.027 [0.000, 0.065] | 0.033 [0.000, 0.083] | 0.367 [0.217, 0.517] | page_then_item / item_only_legacy |
| text/hex | 120 | 120 | 0.990 [0.958, 1.000] | 0.000 [0.000, 0.490] | 1.000 [0.839, 1.000] | 0.000 [0.000, 0.161] | page_then_item / item_only_legacy+wilson_boundary |

Operational/descriptive metrics (retry figures assume IID attempts):

| carrier/encoding | p50/p95 seconds | correct block | C given correct block | attempts/C | input tokens | C/input token |
|---|---:|---:|---:|---:|---:|---:|
| image/hex | 5.0/14.0 | 0.633 | 1.000 | 1.58 | 0 | - |
| text/hex | 5.0/20.0 | 0.990 | 1.000 | 1.01 | 0 | - |

Paired carrier contrasts:

- present_correct: unavailable (no paired items)
- absent_correct: unavailable (no paired items)

Paired architecture contrasts:

- arm_A_minus_B_present_correct_image_hex: unavailable (no paired items)
- arm_A_minus_B_absent_correct_image_hex: unavailable (no paired items)
- arm_B_hex_minus_b32_present_correct: unavailable (no paired items)
