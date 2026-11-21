# EXPLORATORY E4 (already travelled?) — not confirmatory; discovery/validation only; evaluation sealed

Layers [2, 4, 6, 8, 10, 12, 14]; 12 kinds; validation items. Cost ≈ $0.25.

Median over kinds. `clean AUC @L final` = identity readable at the final token at the intervention layer, BEFORE any edit. `r_now` = normalized probe-AUC reduction at layer L, final token; `r_down` = same at L+4.

| condition | L2 | L4 | L6 | L8 | L10 | L12 | L14 |
|---|---|---|---|---|---|---|---|
| clean AUC @L final | 0.923 | 0.818 | 0.705 | 0.805 | 0.736 | 0.756 | 0.788 |
| E4a_spanfinal_meandiff r_now | 1.052 | 0.89 | 1.077 | 0.957 | 1.11 | 0.828 | 0.758 |
| E4a_spanfinal_meandiff r_down | 0.391 | 0.39 | 0.646 | 0.619 | 0.388 | 0.269 | 0.219 |
| E4b_sequence_meandiff r_now | 0.012 | 0.091 | 0.008 | 0.15 | 0.159 | 0.121 | 0.089 |
| E4b_sequence_meandiff r_down | 0.468 | 0.376 | 0.499 | 0.47 | 0.397 | 0.279 | 0.197 |
| E4a_spanfinal_leace r_now | 0.989 | 1.059 | 1.147 | 0.992 | 1.017 | 1.105 | 1.15 |
| E4a_spanfinal_leace r_down | 0.372 | 0.282 | 0.44 | 0.433 | 0.324 | 0.188 | 0.233 |
| E4b_sequence_leace r_now | 0.084 | 0.441 | 0.327 | 0.457 | 0.241 | 0.234 | 0.267 |
| E4b_sequence_leace r_down | 0.499 | 0.255 | 0.422 | 0.396 | 0.254 | 0.161 | 0.152 |
| E4a_spanfinal_randdir r_now | -0.005 | 0.009 | 0.0 | 0.003 | 0.007 | -0.002 | 0.008 |
| E4a_spanfinal_randdir r_down | 0.038 | 0.038 | 0.024 | -0.005 | 0.026 | 0.015 | 0.005 |
| E4b_sequence_randdir r_now | 0.0 | -0.002 | 0.007 | 0.001 | -0.003 | -0.004 | 0.002 |
| E4b_sequence_randdir r_down | 0.011 | 0.003 | 0.006 | 0.033 | 0.0 | 0.019 | 0.005 |
| (E3) E3_meandiff_all r_down | 0.319 | 0.3 | 0.546 | 0.234 | 0.212 | 0.205 | 0.214 |
| (E3) E3_leace_all r_down | 0.246 | 0.242 | 0.363 | 0.157 | 0.158 | 0.125 | 0.145 |
| (E3) E3_randdir_all r_down | -0.003 | 0.024 | 0.007 | 0.031 | 0.01 | 0.008 | 0.007 |
