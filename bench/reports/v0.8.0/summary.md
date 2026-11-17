# Benchmark summary

Numbers are derived from the published run files by summarize.py. Latencies are in milliseconds, TPOT is milliseconds per output token, throughput is in output tokens per second. A dash is a value the file does not hold. The host column names the host whose files a row's numbers come from.

## Gemma

### Gemma A1, 1 in flight (ms, tokens)

| workload | engine | host | N | failed | dropped | e2e p50 | e2e p90 | e2e p99 | e2e mean | TPOT p50 | TPOT p99 | TPOT mean | req/s | tok/s | reply tokens | tokens per verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scene | ours | p1 | 150 | 0 | 0 | 1627.3 | 1778.9 | 1905.4 | 1399.3 | 5.86 | 8.03 | 6.06 | 0.712 | 162.7 | 228.0 | 4.376 |
| scene | vllm | p1 | 150 | 0 | 0 | 1988.7 | 2171.0 | 2229.6 | 1511.5 | 7.07 | 10.85 | 7.95 | 0.660 | 126.7 | 192.0 | 4.325 |
| scene | sglang | p1 | 150 | 0 | 0 | 2524.1 | 3436.8 | 3536.0 | 1972.3 | 12.27 | 17.45 | 13.82 | 0.506 | 79.2 | 157.0 | 4.149 |
| chat | ours | p1 | 80 | 0 | 0 | 3706.3 | 5532.5 | 5769.5 | 3829.6 | 13.97 | 22.06 | 14.45 | 0.261 | 66.7 | 256.0 | 2.036 |
| chat | vllm | p1 | 80 | 0 | 0 | 5070.7 | 5519.2 | 5659.8 | 4307.1 | 19.21 | 21.56 | 16.23 | 0.232 | 59.4 | 256.0 | 1.840 |
| chat | sglang | p1 | 80 | 0 | 0 | 6078.3 | 8396.0 | 8750.1 | 6147.0 | 23.13 | 33.61 | 23.44 | 0.163 | 41.6 | 256.0 | 1.923 |

### Gemma A1, 16 in flight (ms, tokens)

| workload | engine | host | N | failed | dropped | e2e p50 | e2e p90 | e2e p99 | e2e mean | TPOT p50 | TPOT p99 | TPOT mean | req/s | tok/s | reply tokens | tokens per verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scene | ours | p1 | 347 | 0 | 0 | 3207.7 | 3470.9 | 3631.3 | 2771.8 | 11.16 | 18.25 | 11.62 | 5.783 | 1352.9 | 234.0 | 4.383 |
| scene | vllm | p1 | 285 | 0 | 0 | 4315.1 | 4908.7 | 5143.4 | 3385.0 | 15.97 | 31.25 | 17.79 | 4.750 | 935.1 | 197.0 | 4.325 |
| scene | sglang | p1 | 182 | 0 | 0 | 7100.7 | 8477.2 | 8891.0 | 5418.9 | 27.81 | 48.43 | 29.97 | 3.033 | 565.0 | 186.0 | 4.203 |
| chat | ours | p1 | 119 | 0 | 0 | 8411.1 | 10950.9 | 11376.3 | 8263.0 | 32.07 | 43.68 | 31.28 | 1.983 | 507.7 | 256.0 | 1.847 |
| chat | vllm | p1 | 112 | 0 | 0 | 9891.4 | 10692.8 | 11260.9 | 8603.9 | 37.74 | 43.22 | 32.66 | 1.867 | 477.9 | 256.0 | 1.765 |
| chat | sglang | p1 | 72 | 0 | 0 | 15000.2 | 17037.0 | 17564.2 | 13141.3 | 57.72 | 67.77 | 50.26 | 1.200 | 307.2 | 256.0 | 1.809 |

### Gemma A1, 64 in flight (ms, tokens)

| workload | engine | host | N | failed | dropped | e2e p50 | e2e p90 | e2e p99 | e2e mean | TPOT p50 | TPOT p99 | TPOT mean | req/s | tok/s | reply tokens | tokens per verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scene | ours | p1 | 474 | 0 | 0 | 9573.6 | 10405.5 | 10957.2 | 8280.3 | 32.54 | 61.71 | 34.75 | 7.900 | 1834.8 | 232.0 | 4.373 |
| scene | vllm | p1 | 395 | 0 | 0 | 12127.6 | 13676.0 | 14526.0 | 9414.7 | 44.84 | 131.34 | 55.19 | 6.583 | 1282.6 | 195.0 | 4.319 |
| scene | sglang | p1 | 229 | 0 | 0 | 20736.7 | 25056.4 | 26399.0 | 15608.5 | 86.17 | 204.46 | 98.14 | 3.817 | 671.1 | 176.0 | 4.190 |
| chat | ours | p1 | 155 | 0 | 0 | 23957.1 | 30721.9 | 32311.8 | 23661.0 | 77.60 | 107.13 | 76.95 | 2.583 | 661.3 | 256.0 | 1.883 |
| chat | vllm | p1 | 154 | 0 | 0 | 25483.8 | 29496.3 | 31125.3 | 23363.0 | 85.49 | 103.61 | 76.93 | 2.567 | 657.1 | 256.0 | 1.831 |
| chat | sglang | p1 | 74 | 0 | 0 | 44760.6 | 55736.6 | 64899.4 | 44348.9 | 123.79 | 184.92 | 113.10 | 1.233 | 315.7 | 256.0 | 1.801 |

### Gemma bound (from the A1 runs on host p1; ms)

| workload | median of e2e p50 | bound (2 x median) | bound in file | e2e p50 per engine |
|---|---|---|---|---|
| scene | 1988.7 | 3977.4 | 3977.4 | ours 1627.3, vllm 1988.7, sglang 2524.1 |
| chat | 5070.7 | 10141.4 | 10141.4 | ours 3706.3, vllm 5070.7, sglang 6078.3 |

### Gemma A2 highest rate within the bound (requests per second)

| engine | host | best rate | confirmed | first failing | low | high | probes checked | mismatches |
|---|---|---|---|---|---|---|---|---|
| ours | p2 | 3.95 | True | 4.74 | 3.16 | 7.90 | 7 | 0 |
| vllm | p2 | 2.63 | True | 2.96 | 2.63 | 6.58 | 6 | 0 |
| sglang | p2 | none at or above 1.53 | False | 1.53 | 1.53 | 3.82 | 1 | 0 |

### Gemma stability: answers identical between two passes (percent, 95% Wilson interval)

| arm | host | identical/N | % | interval low | interval high | distinct answers p1 / p2 | most common answer p1 / p2 | fp8 kernel | active | status |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | p2 | 500/500 | 100.0 | 99.2 | 100.0 | 416 / 416 | 85 / 85 | CUTLASS | true | ok |
| ours-nobi | p2 | 207/500 | 41.4 | 37.2 | 45.8 | 369 / 370 | 132 / 131 | CUTLASS | false | ok |
| vllm | p2 | 261/500 | 52.2 | 47.8 | 56.5 | 311 / 326 | 190 / 175 | DeepGemmFp8BlockScaledMMKernel | false | ok |
| vllm-bi | p2 | 286/500 | 57.2 | 52.8 | 61.5 | 329 / 330 | 172 / 171 | DeepGemmFp8BlockScaledMMKernel | true | ok |
| vllm-bi-nodg | p3 | 500/500 | 100.0 | 99.2 | 100.0 | 405 / 405 | 96 / 96 | CutlassFp8BlockScaledMMKernel | true | ok |
| sglang | p2 | 335/500 | 67.0 | 62.8 | 71.0 | 289 / 283 | 212 / 218 | auto | false | ok |
| sglang-bi | p2 | 496/500 | 99.2 | 98.0 | 99.7 | 3 / 5 | 498 / 496 | auto | true | ok |

Distinct answers: how many different answer texts (by content digest) a pass gave to its N requests; most common answer: how many requests got the pass's most frequent one. Identical, in the identical/N column, also needs the same token count and finish reason. An identical share means little when a pass gives most requests one answer.

### Gemma cost of invariance at 64 in flight (family I, both modes from fresh servers on one host; other mode = the engine's non-default mode)

| engine | host | source | other mode | workload | status | default tok/s | other tok/s | ratio | default TPOT p50 | other TPOT p50 | default req/s | other req/s | reply tokens default / other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours | p3 | I | ours-nobi | scene | ok | 1848.6 | 1816.7 | 0.983 | 32.45 | 32.58 | 7.967 | 8.350 | 232.0 / 218.0 |
| ours | p3 | I | ours-nobi | chat | ok | 678.4 | 708.3 | 1.044 | 76.45 | 73.87 | 2.650 | 2.767 | 256.0 / 256.0 |
| vllm | p3 | I | vllm-bi-nodg | scene | ok | 1342.7 | 1401.3 | 1.044 | 44.57 | 43.35 | 6.833 | 5.967 | 196.0 / 235.0 |
| vllm | p3 | I | vllm-bi-nodg | chat | ok | 661.3 | 601.6 | 0.910 | 83.30 | 89.26 | 2.583 | 2.350 | 256.0 / 256.0 |
| sglang | p3 | I | sglang-bi | scene | ok | 710.7 | 156.5 | 0.220 | 83.29 | 333.30 | 4.350 | 6.067 | 163.0 / 26.0 |
| sglang | p3 | I | sglang-bi | chat | ok | 328.5 | 174.9 | 0.532 | 100.92 | 204.90 | 1.283 | 0.683 | 256.0 / 256.0 |

vLLM's other mode is `vllm-bi-nodg`, which also turns DeepGEMM off (CUTLASS block FP8): its ratio is the cost of the configuration in which its invariance holds, not of the invariance switch alone.

### vLLM's batch invariance on Gemma 4, by FP8 kernel (percent, 95% Wilson interval)

| arm | host | DeepGEMM | fp8 kernel | active | identical/N | % | interval low | interval high | distinct answers p1 / p2 | most common answer p1 / p2 |
|---|---|---|---|---|---|---|---|---|---|---|
| vllm-bi | p2 | on (vLLM's default) | DeepGemmFp8BlockScaledMMKernel | true | 286/500 | 57.2 | 52.8 | 61.5 | 329 / 330 | 172 / 171 |
| vllm-bi-nodg | p3 | off (VLLM_USE_DEEP_GEMM=0) | CutlassFp8BlockScaledMMKernel | true | 500/500 | 100.0 | 99.2 | 100.0 | 405 / 405 | 96 / 96 |

Distinct answers: how many different answer texts (by content digest) a pass gave to its N requests; most common answer: how many requests got the pass's most frequent one. Identical, in the identical/N column, also needs the same token count and finish reason. An identical share means little when a pass gives most requests one answer.

### Gemma start-up (seconds)

| engine | host | cold ready | cold first answer | warm ready | warm first answer |
|---|---|---|---|---|---|
| ours | p2 | 24.1 | 24.1 | 21.5 | 21.6 |
| vllm | p2 | 203.9 | 205.1 | 68.2 | 68.2 |
| sglang | p2 | 140.8 | 140.9 | 36.4 | 36.5 |

### Gemma validity of structured replies (requests)

| engine | host | requests | stop | valid | invalid | duplicate keys | length | errors |
|---|---|---|---|---|---|---|---|---|
| ours | p2 | 93 | 87 | 87 | 0 | 0 | 6 | 0 |
| vllm | p2 | 93 | 89 | 89 | 0 | 0 | 4 | 0 |
| sglang | p2 | struck (see README) | - | - | - | - | - | - |

### Gemma vLLM's own draft counts (vLLM's settings, not the headline draft count; scene; ms)

| arm | host | status | c=1 TPOT p50 | c=1 e2e p50 | c=1 tokens per verify | c=64 req/s | c=64 TPOT p50 |
|---|---|---|---|---|---|---|---|
| vllm-k1 | p2 | ok | 13.76 | 3904.9 | 1.913 | 5.850 | 51.66 |
| vllm-k4 | p2 | ok | 7.71 | 2143.8 | 3.844 | 6.600 | 46.08 |

## Qwen

### Qwen A1, 1 in flight (ms, tokens)

| workload | engine | host | N | failed | dropped | e2e p50 | e2e p90 | e2e p99 | e2e mean | TPOT p50 | TPOT p99 | TPOT mean | req/s | tok/s | reply tokens | tokens per verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scene | ours | p3 | 150 | 0 | 0 | 258.7 | 281.5 | 1868.8 | 304.6 | 5.91 | 7.86 | 6.00 | 3.257 | 99.0 | 30.0 | 3.622 |
| scene | vllm | p3 | 150 | 0 | 0 | 409.1 | 440.9 | 2667.3 | 467.8 | 9.90 | 12.19 | 10.01 | 2.128 | 63.1 | 30.0 | 3.227 |
| scene | sglang | p3 | 150 | 0 | 0 | 417.9 | 447.3 | 2649.5 | 489.6 | 9.69 | 11.37 | 9.78 | 2.032 | 64.0 | 32.0 | 3.247 |
| chat | ours | p3 | 80 | 0 | 0 | 2013.3 | 3086.2 | 3467.6 | 2166.4 | 7.45 | 13.16 | 8.05 | 0.461 | 118.0 | 256.0 | 2.985 |
| chat | vllm | p3 | 80 | 0 | 0 | 2578.4 | 3778.4 | 4539.0 | 2760.9 | 9.57 | 17.36 | 10.31 | 0.362 | 92.7 | 256.0 | 2.922 |
| chat | sglang | p3 | 80 | 0 | 0 | 2559.8 | 3987.3 | 4278.5 | 2752.0 | 9.47 | 16.22 | 10.25 | 0.363 | 92.9 | 256.0 | 2.844 |

### Qwen A1, 16 in flight (ms, tokens)

| workload | engine | host | N | failed | dropped | e2e p50 | e2e p90 | e2e p99 | e2e mean | TPOT p50 | TPOT p99 | TPOT mean | req/s | tok/s | reply tokens | tokens per verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scene | ours | p3 | 501 | 0 | 0 | 1779.4 | 2068.8 | 8445.4 | 1922.9 | 51.09 | 67.93 | 50.90 | 8.350 | 218.2 | 26.0 | 3.719 |
| scene | vllm | p3 | 371 | 0 | 0 | 2358.0 | 2651.6 | 16600.5 | 2562.4 | 63.33 | 91.71 | 62.76 | 6.183 | 165.8 | 27.0 | 3.227 |
| scene | sglang | p3 | 348 | 0 | 0 | 2363.1 | 2650.4 | 22729.9 | 2836.2 | 34.33 | 106.98 | 42.69 | 5.800 | 172.5 | 30.0 | 3.227 |
| chat | ours | p3 | 229 | 0 | 0 | 4028.0 | 5875.7 | 6685.7 | 4204.4 | 14.95 | 25.55 | 15.67 | 3.817 | 977.1 | 256.0 | 2.695 |
| chat | vllm | p3 | 181 | 0 | 0 | 5001.6 | 7158.7 | 7770.6 | 5219.0 | 18.79 | 29.69 | 19.56 | 3.017 | 772.3 | 256.0 | 2.662 |
| chat | sglang | p3 | 187 | 0 | 0 | 4928.3 | 7159.5 | 8047.6 | 5145.3 | 18.41 | 30.83 | 19.34 | 3.117 | 797.9 | 256.0 | 2.662 |

### Qwen A1, 64 in flight (ms, tokens)

| workload | engine | host | N | failed | dropped | e2e p50 | e2e p90 | e2e p99 | e2e mean | TPOT p50 | TPOT p99 | TPOT mean | req/s | tok/s | reply tokens | tokens per verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scene | ours | p3 | 509 | 0 | 0 | 7246.8 | 7534.4 | 25617.3 | 7520.4 | 68.29 | 88.05 | 67.42 | 8.483 | 234.6 | 28.0 | 3.677 |
| scene | vllm | p3 | 387 | 0 | 0 | 9167.6 | 10094.8 | 48329.0 | 10242.1 | 287.30 | 342.91 | 282.74 | 6.450 | 178.2 | 28.0 | 3.203 |
| scene | sglang | p3 | 358 | 0 | 0 | 9107.7 | 11358.5 | 60066.3 | 10639.7 | 213.86 | 304.33 | 213.03 | 5.967 | 186.9 | 31.0 | 3.190 |
| chat | ours | p3 | 358 | 0 | 0 | 10112.3 | 14830.2 | 17616.6 | 10885.6 | 34.81 | 63.58 | 37.35 | 5.967 | 1527.5 | 256.0 | 2.891 |
| chat | vllm | p3 | 291 | 0 | 0 | 12371.8 | 18942.1 | 21520.4 | 13408.6 | 46.54 | 81.96 | 50.48 | 4.850 | 1241.6 | 256.0 | 2.843 |
| chat | sglang | p3 | 302 | 0 | 0 | 11929.2 | 17463.6 | 19650.5 | 12947.7 | 33.51 | 63.33 | 37.22 | 5.033 | 1288.5 | 256.0 | 2.836 |

SGLang ran at 48 running requests (a named deviation: its fp32 mamba state does not fit 64), so at 64 in flight 16 requests queue in its server.

### Qwen bound (from the A1 runs on host p2; ms)

| workload | median of e2e p50 | bound (2 x median) | bound in file | e2e p50 per engine |
|---|---|---|---|---|
| scene | 333.8 | 667.6 | 667.6 | ours 258.5, vllm 409.1 |
| chat | 2325.6 | 4651.2 | 4651.2 | ours 2014.0, vllm 2637.2 |

### Qwen A2 highest rate within the bound (requests per second)

| engine | host | best rate | confirmed | first failing | low | high | probes checked | mismatches |
|---|---|---|---|---|---|---|---|---|
| ours | p2 | none at or above 3.42 | False | 3.42 | 3.42 | 8.55 | 1 | 0 |
| vllm | p2 | none at or above 2.59 | False | 2.59 | 2.59 | 6.48 | 1 | 0 |
| sglang | - | not run | - | - | - | - | - | - |

### Qwen stability: answers identical between two passes (percent, 95% Wilson interval)

| arm | host | identical/N | % | interval low | interval high | distinct answers p1 / p2 | most common answer p1 / p2 | fp8 kernel | active | status |
|---|---|---|---|---|---|---|---|---|---|---|
| ours | p2 | 496/500 | 99.2 | 98.0 | 99.7 | 17 / 16 | 378 / 376 | CUTLASS | true | ok |
| ours-nobi | p2 | 387/500 | 77.4 | 73.5 | 80.8 | 15 / 14 | 397 / 411 | CUTLASS | false | ok |
| vllm | p2 | 326/500 | 65.2 | 60.9 | 69.2 | 16 / 14 | 308 / 353 | CutlassFp8BlockScaledMMKernel | false | ok |
| vllm-bi | p2 | - | - | - | - | - | - | - | not recorded | unsupported |
| sglang | p3 | 321/500 | 64.2 | 59.9 | 68.3 | 19 / 18 | 345 / 335 | auto | false | ok |
| sglang-bi | p3 | 320/500 | 64.0 | 59.7 | 68.1 | 20 / 13 | 321 / 316 | auto | true | ok |

Distinct answers: how many different answer texts (by content digest) a pass gave to its N requests; most common answer: how many requests got the pass's most frequent one. Identical, in the identical/N column, also needs the same token count and finish reason. An identical share means little when a pass gives most requests one answer.

### Qwen cost of invariance at 64 in flight (A1 against the C pass-1 load of the other mode, both from one host; other mode = the engine's non-default mode)

| engine | host | source | other mode | workload | status | default tok/s | other tok/s | ratio | default TPOT p50 | other TPOT p50 | default req/s | other req/s | reply tokens default / other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours | p2 | A1 vs C pass 1 | ours-nobi | scene | ok | 236.1 | 239.2 | 1.013 | 67.84 | 66.74 | 8.550 | 8.733 | 28.0 / 27.0 |
| ours | p2 | A1 vs C pass 1 | ours-nobi | chat | ok | 1527.5 | 1514.7 | 0.992 | 34.38 | 35.35 | 5.967 | 5.917 | 256.0 / 256.0 |
| vllm | p2 | A1 vs C pass 1 | vllm-bi | scene | unsupported | - | - | - | - | - | - | - | - |
| vllm | p2 | A1 vs C pass 1 | vllm-bi | chat | unsupported | - | - | - | - | - | - | - | - |
| sglang | p3 | A1 vs C pass 1 | sglang-bi | scene | ok | 186.9 | 170.4 | 0.912 | 213.86 | 246.92 | 5.967 | 5.700 | 31.0 / 30.0 |
| sglang | p3 | A1 vs C pass 1 | sglang-bi | chat | ok | 1288.5 | 955.7 | 0.742 | 33.51 | 44.89 | 5.033 | 3.733 | 256.0 / 256.0 |

SGLang's other mode on Qwen3.8 also runs the triton attention backend (flashinfer in its default mode): its ratio includes that change.

### Qwen start-up (seconds)

| engine | host | cold ready | cold first answer | warm ready | warm first answer |
|---|---|---|---|---|---|
| ours | p2 | 27.6 | 27.7 | 24.8 | 24.9 |
| vllm | p2 | 212.3 | 212.4 | 75.1 | 75.2 |
| sglang | p3 | 155.1 | 155.3 | 61.7 | 61.8 |

### Qwen validity of structured replies (requests)

| engine | host | requests | stop | valid | invalid | duplicate keys | length | errors |
|---|---|---|---|---|---|---|---|---|
| ours | p2 | 93 | 92 | 91 | 0 | 1 | 1 | 0 |
| vllm | p2 | 93 | 90 | 86 | 0 | 4 | 3 | 0 |
| sglang | p3 | not run (no structured-output row for SGLang) | - | - | - | - | - | - |

### Qwen vLLM's own draft counts (vLLM's settings, not the headline draft count; scene; ms)

| arm | host | status | c=1 TPOT p50 | c=1 e2e p50 | c=1 tokens per verify | c=64 req/s | c=64 TPOT p50 |
|---|---|---|---|---|---|---|---|
| vllm-k1 | p2 | ok | 13.99 | 501.1 | 1.902 | 6.567 | 312.97 |
| vllm-k4 | - | not run | - | - | - | - | - |

## Vendor-image check

### Gemma scene at one request in flight (hosts q-vllm and q-sglang against p1; TPOT in ms; pass when |deviation| <= 5%)

| engine | status | q TPOT p50 | p1 TPOT p50 | host ratio | deviation | verdict |
|---|---|---|---|---|---|---|
| vllm | ok | 7.15 | 7.07 | 0.9979 | -1.32% | pass |
| sglang | ok | 12.72 | 12.27 | 0.9845 | -5.04% | fail |

## GPU controls

### Start, middle and end controls (TFLOPS, GB/s, MHz, W; SM clock and power: bf16 phase medians)

| control | bf16 TFLOPS | fp8 TFLOPS | copy GB/s | SM MHz | power W |
|---|---|---|---|---|---|
| p1-start | 402.7 | 758.7 | 1459.1 | 2160 | 600.0 |
| p2-end | 405.8 | 772.0 | 1460.2 | 2175 | 600.0 |
| p2-middle | 410.8 | 776.4 | 1460.1 | 2205 | 600.0 |
| p2-start | 410.9 | 777.9 | 1459.8 | 2212 | 600.0 |
| p3-end | 400.5 | 761.8 | 1459.9 | 2145 | 600.0 |
| p3-middle | 406.7 | 771.4 | 1459.9 | 2182 | 600.0 |
| p3-start | 411.2 | 781.3 | 1460.5 | 2220 | 600.0 |
| q-sglang-end | 402.8 | 767.9 | 1459.8 | 2160 | 600.0 |
| q-sglang-start | 408.3 | 772.1 | 1460.0 | 2190 | 600.0 |
| q-vllm-end | 398.0 | 760.2 | 1460.0 | 2130 | 600.0 |
| q-vllm-start | 402.3 | 762.6 | 1460.0 | 2160 | 600.0 |

## llm-mini-infer's grammar compile cost (CPU, host p2)

### Gemma (ms unless the name says otherwise)

| field | value |
|---|---|
| model | "gemma-4-31b-it" |
| vocab | 262144 |
| backend | "guidance" |
| compact | true |
| llguidance | "1.9.1" |
| k | 5 |
| tokenizer_build_s | 2.141 |
| compile_corpus_ms_p50_max | [0.386, 3.593] |
| compile_json_object_ms | 0.352 |
| compile_catalog_cold_ms | 0.664 |
| compile_catalog_again_ms | [0.577, 0.538, 0.509, 0.494, 0.484] |
| clone_us_p50 | 0.0 |

Mask times, p50 / p99 (ms):

| case | 6_masks_ms | one_mask_ms |
|---|---|---|
| threads=1 rows=1 | 0.037 / 0.179 | 0.010 / 0.090 |
| threads=1 rows=8 | 0.286 / 0.564 | 0.060 / 0.266 |
| threads=1 rows=32 | 1.069 / 1.547 | 0.207 / 0.395 |
| threads=1 rows=64 | 2.120 / 2.653 | 0.425 / 0.545 |
| threads=2 rows=1 | 0.030 / 0.074 | 0.006 / 0.057 |
| threads=2 rows=8 | 0.338 / 0.686 | 0.060 / 0.265 |
| threads=2 rows=32 | 1.219 / 1.581 | 0.232 / 0.429 |
| threads=2 rows=64 | 2.517 / 3.085 | 0.472 / 0.673 |
| threads=4 rows=1 | 0.028 / 0.105 | 0.006 / 0.477 |
| threads=4 rows=8 | 0.412 / 0.777 | 0.098 / 0.372 |
| threads=4 rows=32 | 1.540 / 2.593 | 0.314 / 0.892 |
| threads=4 rows=64 | 3.026 / 4.263 | 0.623 / 0.847 |
| threads=8 rows=1 | 0.037 / 0.103 | 0.009 / 0.076 |
| threads=8 rows=8 | 0.441 / 1.345 | 0.122 / 0.447 |
| threads=8 rows=32 | 1.562 / 2.018 | 0.334 / 0.496 |
| threads=8 rows=64 | 2.943 / 3.370 | 0.610 / 0.802 |

### Qwen (ms unless the name says otherwise)

| field | value |
|---|---|
| model | "qwen3.8-27b" |
| vocab | 248320 |
| backend | "guidance" |
| compact | true |
| llguidance | "1.9.1" |
| k | 3 |
| tokenizer_build_s | 1.308 |
| compile_corpus_ms_p50_max | [0.377, 1.229] |
| compile_json_object_ms | 0.372 |
| compile_catalog_cold_ms | 0.581 |
| compile_catalog_again_ms | [0.526, 0.491, 0.478, 0.47, 0.471] |
| clone_us_p50 | 1.0 |

Mask times, p50 / p99 (ms):

| case | 4_masks_ms | one_mask_ms |
|---|---|---|
| threads=1 rows=1 | 0.026 / 0.061 | 0.011 / 0.517 |
| threads=1 rows=8 | 0.213 / 0.569 | 0.065 / 0.424 |
| threads=1 rows=32 | 0.828 / 1.044 | 0.212 / 0.379 |
| threads=1 rows=64 | 1.653 / 1.958 | 0.477 / 0.594 |
| threads=2 rows=1 | 0.024 / 0.074 | 0.008 / 0.080 |
| threads=2 rows=8 | 0.205 / 0.554 | 0.063 / 0.375 |
| threads=2 rows=32 | 0.736 / 0.984 | 0.208 / 0.345 |
| threads=2 rows=64 | 1.352 / 1.722 | 0.394 / 0.537 |
| threads=4 rows=1 | 0.024 / 0.058 | 0.010 / 0.106 |
| threads=4 rows=8 | 0.227 / 0.470 | 0.071 / 0.270 |
| threads=4 rows=32 | 0.868 / 1.227 | 0.281 / 0.409 |
| threads=4 rows=64 | 1.703 / 2.068 | 0.500 / 0.639 |
| threads=8 rows=1 | 0.023 / 0.049 | 0.008 / 0.097 |
| threads=8 rows=8 | 0.260 / 0.599 | 0.077 / 0.398 |
| threads=8 rows=32 | 0.953 / 1.264 | 0.284 / 0.500 |
| threads=8 rows=64 | 1.830 / 2.197 | 0.560 / 0.717 |

## Clock check: loads more than 3% from their host's median SM clock

### Host p1 (median of 18 loads: 2388 MHz)

| unit | load | SM MHz | diff % | power W | throttle |
|---|---|---|---|---|---|
| gemma/A1/ours | chat-c64 | 2310 | -3.29 | 574.7 | [4] |

| control | bf16 SM MHz | bf16 TFLOPS |
|---|---|---|
| start | 2160 | 402.7 |

### Host p2 (median of 53 loads: 2377 MHz)

| unit | load | SM MHz | diff % | power W | throttle |
|---|---|---|---|---|---|
| qwen/A1/ours | scene-c64 | 2302 | -3.16 | 600.3 | [4] |
| qwen/A1/vllm | scene-c16 | 2295 | -3.45 | 596.0 | [4] |
| qwen/A1/vllm | scene-c64 | 2257 | -5.05 | 600.1 | [4] |
| qwen/B/vllm-k1 | scene-c64 | 2272 | -4.42 | 599.6 | [4] |
| qwen/C/ours-nobi-p1 | stability | 2302 | -3.16 | 598.6 | [4] |
| qwen/C/ours-nobi-p1 | scene-c64 | 2272 | -4.42 | 599.6 | [4] |
| qwen/C/ours-nobi-p1 | chat-c64 | 2295 | -3.45 | 599.2 | [4] |
| qwen/C/ours-nobi-p2 | stability | 2302 | -3.16 | 598.3 | [4] |
| qwen/C/vllm-p2 | stability | 2302 | -3.16 | 592.4 | [4] |

| control | bf16 SM MHz | bf16 TFLOPS |
|---|---|---|
| start | 2212 | 410.9 |
| middle | 2205 | 410.8 |
| end | 2175 | 405.8 |

### Host p3 (median of 40 loads: 2366 MHz)

| unit | load | SM MHz | diff % | power W | throttle |
|---|---|---|---|---|---|
| qwen/A1/ours | scene-c64 | 2287 | -3.34 | 600.0 | [4] |
| qwen/A1/vllm | scene-c16 | 2295 | -3.00 | 597.5 | [4] |
| qwen/A1/vllm | scene-c64 | 2238 | -5.39 | 599.8 | [4] |

| control | bf16 SM MHz | bf16 TFLOPS |
|---|---|---|
| start | 2220 | 411.2 |
| middle | 2182 | 406.7 |
| end | 2145 | 400.5 |

### Host q-sglang (median of 2 loads: 2407 MHz)

No load differs by more than 3%.

| control | bf16 SM MHz | bf16 TFLOPS |
|---|---|---|
| start | 2190 | 408.3 |
| end | 2160 | 402.8 |

### Host q-vllm (median of 2 loads: 2384 MHz)

No load differs by more than 3%.

| control | bf16 SM MHz | bf16 TFLOPS |
|---|---|---|
| start | 2160 | 402.3 |
| end | 2130 | 398.0 |

## Manifests whose status is not ok

| manifest | status | status line |
|---|---|---|
| p2/qwen/A1/sglang | failed | ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.7971479687499999. Raise -- |
| p2/qwen/C/vllm-bi-p1 | unsupported | RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN. |
| p2/qwen/C/vllm-bi-p2 | unsupported | RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN. |
| p2/qwen/smoke/sglang-bi | failed | ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.7971479687499999. Raise -- |
| p2/qwen/smoke/sglang | failed | ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.85. Raise --mem-fraction-s |
| p2/qwen/smoke/vllm-bi | unsupported | RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN. |

