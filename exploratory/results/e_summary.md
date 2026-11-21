# EXPLORATORY E1-E3 — not confirmatory; discovery/validation only; evaluation sealed

Layers 2–14, 12 kinds, validation items. Cost ≈ $1.43 (runtime 78 GPU-min).

| condition | met R1 bar (of 156) | median r (IQR) | max r | median amnesic AUC | median role drop | median ‖Δ‖/‖h‖ | median KL |
|---|---|---|---|---|---|---|---|
| E1_k8_last | 0 | 0.0 (-0.005, 0.006) | 0.144 | 0.989 | 0.0 | 0.053 | 0.0 |
| RAND_k8_last | 0 | 0.0 (-0.003, 0.002) | 0.148 | 0.989 | 0.0 | 0.037 | 0.0 |
| E1_k8_all | 0 | 0.002 (-0.008, 0.013) | 0.144 | 0.989 | 0.0 | 0.053 | 0.0 |
| RAND_k8_all | 0 | 0.0 (-0.007, 0.003) | 0.185 | 0.989 | 0.0 | 0.037 | 0.0 |
| E1_k16_last | 0 | 0.0 (-0.006, 0.008) | 0.144 | 0.989 | 0.0 | 0.058 | 0.0 |
| RAND_k16_last | 0 | 0.0 (-0.006, 0.005) | 0.237 | 0.989 | 0.0 | 0.07 | 0.0 |
| E1_k16_all | 0 | 0.003 (-0.006, 0.013) | 0.144 | 0.989 | 0.0 | 0.058 | 0.0 |
| RAND_k16_all | 0 | 0.0 (-0.01, 0.007) | 0.23 | 0.989 | 0.0 | 0.07 | 0.0 |
| E1_k32_last | 0 | 0.0 (-0.005, 0.008) | 0.144 | 0.989 | 0.0 | 0.059 | 0.0 |
| RAND_k32_last | 0 | 0.0 (-0.009, 0.011) | 0.259 | 0.991 | 0.0 | 0.115 | 0.0 |
| E1_k32_all | 0 | 0.003 (-0.006, 0.013) | 0.155 | 0.989 | 0.0 | 0.059 | 0.0 |
| RAND_k32_all | 0 | 0.0 (-0.014, 0.013) | 0.267 | 0.991 | 0.0 | 0.115 | 0.001 |
| E1_k64_last | 0 | 0.0 (-0.005, 0.008) | 0.144 | 0.99 | 0.0 | 0.059 | 0.0 |
| RAND_k64_last | 0 | 0.002 (-0.014, 0.016) | 0.178 | 0.99 | 0.0 | 0.196 | 0.0 |
| E1_k64_all | 0 | 0.004 (-0.006, 0.014) | 0.155 | 0.99 | 0.0 | 0.059 | 0.0 |
| RAND_k64_all | 0 | 0.004 (-0.021, 0.022) | 0.235 | 0.99 | 0.0 | 0.196 | 0.002 |
| E1_k128_last | 0 | 0.0 (-0.005, 0.009) | 0.144 | 0.99 | 0.0 | 0.064 | 0.0 |
| RAND_k128_last | 0 | 0.009 (-0.013, 0.04) | 0.338 | 0.99 | 0.0 | 0.302 | 0.001 |
| E1_k128_all | 0 | 0.005 (-0.006, 0.014) | 0.155 | 0.99 | 0.0 | 0.064 | 0.0 |
| RAND_k128_all | 0 | 0.015 (-0.019, 0.069) | 0.488 | 0.99 | 0.0 | 0.302 | 0.004 |
| E3_meandiff_last | 2 | 0.206 (0.131, 0.303) | 0.837 | 0.5 | 0.0 | 0.2 | 0.001 |
| E3_leace_last | 0 | 0.118 (0.079, 0.168) | 0.612 | 0.5 | 0.0 | 0.144 | 0.0 |
| E3_randdir_last | 0 | 0.011 (-0.001, 0.032) | 0.5 | 0.987 | 0.005 | 0.131 | 0.0 |
| E3_meandiff_all | 7 | 0.261 (0.166, 0.346) | 1.0 | 0.5 | 0.0 | 0.2 | 0.002 |
| E3_leace_all | 0 | 0.18 (0.116, 0.271) | 0.625 | 0.5 | 0.0 | 0.144 | 0.001 |
| E3_randdir_all | 0 | 0.013 (0.0, 0.053) | 0.525 | 0.987 | 0.005 | 0.131 | 0.001 |

## Conditions meeting the R1 bar (descriptive)

- L2 aluminium beverage can E3_meandiff_last: r = 0.837 (95% CI [0.19, 8.69]), role drop -0.017
- L2 aluminium beverage can E3_meandiff_all: r = 1.0 (95% CI [0.23, 10.66]), role drop -0.017
- L2 parchment E3_meandiff_all: r = 0.72 (95% CI [0.36, 1.67]), role drop 0.01
- L2 wristwatch E3_meandiff_all: r = 0.677 (95% CI [0.27, 1.78]), role drop 0.031
- L5 car windshield E3_meandiff_all: r = 0.683 (95% CI [0.4, 1.25]), role drop 0.01
- L6 acrylic sheet E3_meandiff_last: r = 0.77 (95% CI [0.23, 7.75]), role drop -0.003
- L6 balsa strip E3_meandiff_all: r = 0.67 (95% CI [0.27, 5.89]), role drop 0.005
- L6 car windshield E3_meandiff_all: r = 0.701 (95% CI [0.39, 1.53]), role drop 0.021
- L6 raw egg white E3_meandiff_all: r = 0.74 (95% CI [0.37, 1.95]), role drop 0.016

## Exploratory D comparison (discovery causal items, eligible only)

- L2 aluminium beverage can E3_meandiff_last: n = 0 items / 0 events; Δ1 = None, Δ2 = None
- L2 aluminium beverage can E3_meandiff_all: n = 0 items / 0 events; Δ1 = None, Δ2 = None
- L2 parchment E3_meandiff_all: n = 0 items / 0 events; Δ1 = None, Δ2 = None
- L5 car windshield E3_meandiff_all: n = 5 items / 5 events; Δ1 = 0.024235930930610515, Δ2 = 0.09149851840069982
- L6 acrylic sheet E3_meandiff_last: n = 3 items / 3 events; Δ1 = -0.0697233385954273, Δ2 = -0.027650642887009063
- L6 balsa strip E3_meandiff_all: n = 0 items / 0 events; Δ1 = None, Δ2 = None
- L6 car windshield E3_meandiff_all: n = 5 items / 5 events; Δ1 = 0.05842864412123938, Δ2 = 0.10467736040800091
- L6 raw egg white E3_meandiff_all: n = 0 items / 0 events; Δ1 = None, Δ2 = None
